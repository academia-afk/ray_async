import os
import json
import random
import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import gym
import wandb
import ray
from ray.tune import register_env
from pycocotools.cocoeval import COCOeval




def collate_fn(batch):
    return tuple(zip(*batch))


class EnvClass(gym.Env):
    def __init__(self, seed):
        self.seed(seed)


class CocoDataset(CocoDetection):
    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        img = T.ToTensor()(img)

        coco_img_id = self.ids[idx]
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([coco_img_id], dtype=torch.int64)
        }
        return img, target


def evaluate_coco(model, data_loader, device, dataset_dir):
    model.eval()
    coco_gt = data_loader.dataset.coco
    results = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = target["image_id"].item()
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score)
                    })

    pred_file = os.path.join(dataset_dir, "predictions.json")
    with open(pred_file, "w") as f:
        json.dump(results, f, indent=2)

    coco_dt = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap = coco_eval.stats[0]
    ap50 = coco_eval.stats[1]
    return ap, ap50


@ray.remote
class ParameterServer:
    def __init__(self, initial_weights, lr=0.005):
        # Store all initial weights on CPU
        self.global_weights = {k: v.cpu() for k, v in initial_weights.items()}
        self.lr = lr

    def get_weights(self):
        return self.global_weights

    def apply_gradients(self, grad_dict):
        # Only update keys that exist in grad_dict
        for key, grad in grad_dict.items():
            self.global_weights[key] = self.global_weights[key] - self.lr * grad

    def set_lr(self, new_lr):
        self.lr = new_lr
        

@ray.remote(num_gpus=1)
def train_loop_per_worker(worker_id, config, ps_actor):
    def set_seed(seed=40):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    set_seed(40)

    train_dir = config["train_dir"]
    val_dir   = config["val_dir"]

    train_dataset = CocoDataset(
        os.path.join(train_dir, "data"),
        os.path.join(train_dir, "labels.json")
    )
    val_dataset = CocoDataset(
        os.path.join(val_dir, "data"),
        os.path.join(val_dir, "labels.json")
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=config["num_nodes"],
        rank=worker_id,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = base_model.roi_heads.box_predictor.cls_score.in_features
    base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config["num_classes"])

    for param in base_model.backbone.parameters():
        param.requires_grad = False
    for param in base_model.rpn.parameters():
        param.requires_grad = False

    base_model.to(device)

    optimizer = optim.SGD(
        [p for p in base_model.parameters() if p.requires_grad],
        lr=(config["lr"] * config["num_nodes"]),
        momentum=0.9, weight_decay=0.0005
    )

    wandb.init(
        project="async_distributed",
        group="five_nodes",
        name=f"worker_{worker_id}",
        config=config
    )

    num_epochs = config["num_epochs"]
    for epoch in range(num_epochs):
        base_model.train()
        train_sampler.set_epoch(epoch)

        total_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            latest_weights = ray.get(ps_actor.get_weights.remote())
            for name, param in base_model.state_dict().items():
                updated_val = latest_weights[name]
                if isinstance(updated_val, torch.Tensor):
                    updated_val = updated_val.to(device)
                base_model.state_dict()[name].copy_(updated_val)

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = base_model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()

            total_loss += loss.item()


            grad_dict = {}
            for name, param in base_model.named_parameters():
                if param.grad is not None:
                    grad_dict[name] = param.grad.detach().cpu()

            ps_actor.apply_gradients.remote(grad_dict)

        avg_loss = total_loss / len(train_loader)

        if worker_id == 0:
            latest_weights = ray.get(ps_actor.get_weights.remote())
            for name, param in base_model.state_dict().items():
                updated_val = latest_weights[name].to(device)
                param.copy_(updated_val)

            ap, ap50 = evaluate_coco(base_model, val_loader, device, val_dir)
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "ap": ap,
                "ap50": ap50,
            })
            print(f"[Worker {worker_id}] Epoch {epoch}: loss={avg_loss:.4f}, AP={ap:.4f}, AP50={ap50:.4f}")
        else:
            wandb.log({"epoch": epoch, "loss": avg_loss})

    wandb.finish()


    final_weights = ray.get(ps_actor.get_weights.remote())
    for name, param in base_model.state_dict().items():
        param.copy_(final_weights[name].to(device))

    return f"Worker {worker_id} done."


if __name__ == "__main__":
    ray.init(address="auto") 

    register_env("my_seeded_env", lambda c: EnvClass(c))

    config = {
        "train_dir": "/workspace/dataset/training",  
        "val_dir":   "/workspace/dataset/validation", 
        "num_classes": 20,
        "num_nodes": 5,        
        "batch_size": 8,
        "num_epochs": 10,
        "lr": 0.005,      
    }
    initial_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = initial_model.roi_heads.box_predictor.cls_score.in_features
    initial_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config["num_classes"])

    for param in initial_model.backbone.parameters():
        param.requires_grad = False
    for param in initial_model.rpn.parameters():
        param.requires_grad = False

    initial_weights = initial_model.state_dict()

    ps = ParameterServer.options(name="ps-actor").remote(
        initial_weights=initial_weights,
        lr=config["lr"]
    )

    futures = [
        train_loop_per_worker.remote(
            worker_id=i,
            config=config,
            ps_actor=ps
        )
        for i in range(config["num_nodes"])
    ]

    results = ray.get(futures)
    print("RESULTS:", results)

    ray.shutdown()
