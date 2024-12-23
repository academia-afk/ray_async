import json
import os
import random

import gym
import numpy as np
import ray
import torch
import torchvision
import torchvision.transforms as T
import wandb
from pycocotools.cocoeval import COCOeval
from ray.tune import register_env
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



class EnvClass(gym.Env):
    """Minimal gym Env to keep Ray Tune from complaining."""
    def __init__(self, seed):
        self.seed(seed)



class CocoDataset(CocoDetection):
    """Standard COCO detection dataset wrapper."""
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


def collate_fn(batch):
    return tuple(zip(*batch))


def evaluate_coco(model, data_loader, device, dataset_dir):
    """Compute COCO AP on a validation set."""
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


@ray.remote(num_gpus=1)
def train_one_epoch(
    node_id: int,
    config: dict,
    initial_weights: dict
):

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(40)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dir = config["train_dir"]
    val_dir   = config["val_dir"]

    train_dataset = CocoDataset(
        os.path.join(train_dir, "data"),
        os.path.join(train_dir, "labels.json")
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=config["num_nodes"],
        rank=node_id,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        collate_fn=collate_fn
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config["num_classes"])


    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.rpn.parameters():
        param.requires_grad = False

    model.to(device)

    model_state = model.state_dict()
    for k, v in initial_weights.items():
        model_state[k].copy_(v.to(device))
    model.load_state_dict(model_state)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"] * config["num_nodes"],
        momentum=0.9,
        weight_decay=0.0005,
    )

    wandb.init(
        project="sync_distributed",
        name=f"worker_{node_id}",
        group="three_nodes",
        config=config
    )

    model.train()
    total_train_loss = 0.0

    train_sampler.set_epoch(random.randint(0, 9999))  # shuffle
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_loss = total_train_loss / len(train_loader)

    wandb.log({"loss": avg_loss})
    wandb.finish()

    final_state = model.state_dict()
    for k, v in final_state.items():
        final_state[k] = v.cpu()

    return {
        "worker_id": node_id,
        "avg_loss": avg_loss,
        "model_state": final_state
    }


def average_model_states(list_of_states):
    num = len(list_of_states)
    merged = {}
    for key in list_of_states[0].keys():
        merged[key] = sum(state[key] for state in list_of_states) / num
    return merged


if __name__ == "__main__":
    import math
    ray.init(address="auto")
    register_env("my_seeded_env", lambda c: EnvClass(c))

    config = {
        "train_dir":   "/workspace/dataset/training",
        "val_dir":     "/workspace/dataset/validation",
        "num_classes": 20,
        "num_nodes":   3,
        "batch_size":  8,
        "num_epochs":  10,
        "lr":          0.005,
    }

    base_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = base_model.roi_heads.box_predictor.cls_score.in_features
    base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config["num_classes"])

    for param in base_model.backbone.parameters():
        param.requires_grad = False
    for param in base_model.rpn.parameters():
        param.requires_grad = False

    global_state = base_model.state_dict()
    for k, v in global_state.items():
        global_state[k] = v.cpu()

    wandb.init(project="sync_distributed", group="five_nodes", name="driver", config=config)

    for epoch in range(config["num_epochs"]):
        futures = []
        for node_id in range(config["num_nodes"]):
            fut = train_one_epoch.remote(
                node_id=node_id,
                config=config,
                initial_weights=global_state
            )
            futures.append(fut)

        results = ray.get(futures)

        states_to_merge = [res["model_state"] for res in results]
        global_state = average_model_states(states_to_merge)

        all_losses = [res["avg_loss"] for res in results]
        epoch_avg_loss = sum(all_losses) / len(all_losses)

        base_model.load_state_dict(global_state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model.to(device)

        ap, ap50 = evaluate_coco(base_model, 
                                 DataLoader(
                                     CocoDataset(
                                         os.path.join(config["val_dir"], "data"),
                                         os.path.join(config["val_dir"], "labels.json")
                                     ),
                                     batch_size=config["batch_size"],
                                     shuffle=False,
                                     collate_fn=collate_fn
                                 ),
                                 device,
                                 config["val_dir"]
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_avg_loss,
            "ap": ap,
            "ap50": ap50,
        })

        print(
            f"Epoch [{epoch+1}/{config['num_epochs']}], "
            f"avg_loss={epoch_avg_loss:.4f}, AP={ap:.4f}, AP50={ap50:.4f}"
        )

    torch.save(global_state, f"naive_sync_{config['num_nodes']}_workers.pth")

    wandb.finish()
    ray.shutdown()
