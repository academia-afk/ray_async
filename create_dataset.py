import fiftyone as fo
import fiftyone.zoo as foz
import os

classes = ["Cat", "Dog", "Person"]
max_samples_train = 80
max_samples_val = 20

export_dir = "/workspace/dataset"

os.makedirs(export_dir, exist_ok=True)

train_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=classes,
    max_samples=max_samples_train,
    only_matching=True,
)
train_export_dir = os.path.join(export_dir, "training")
train_dataset.export(
    export_dir=train_export_dir,
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
)

val_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=classes,
    max_samples=max_samples_val,
    only_matching=True,
)
val_export_dir = os.path.join(export_dir, "validation")
val_dataset.export(
    export_dir=val_export_dir,
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
)

del train_dataset
del val_dataset
