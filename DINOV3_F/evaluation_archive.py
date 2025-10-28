import os
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Config
DATA_ROOT = Path("/mnt/d/Project/archive")
NUM_CLASSES = 4
REPO_DIR = Path("/home/phil/dinov3")
LOCAL_WEIGHTS = "/mnt/d/Project/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
CKPT_DIR = Path("dinov3_archive_checkpoints")
CHECKPOINT_PATH = CKPT_DIR / "best_model.pt"

print(f"Using device: {DEVICE}")

# Transforms
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load test dataset
def load_split(split_name, transform):
    split_path = DATA_ROOT / split_name
    dataset = datasets.ImageFolder(root=split_path, transform=transform)
    print(f"{split_name.capitalize()} dataset: {len(dataset)} samples")
    print(f"{split_name.capitalize()} class distribution: {Counter(dataset.targets)}")
    return dataset

test_dataset = load_split("test", test_transform)
classes = test_dataset.classes
print(f"Classes: {classes}")

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Model load
backbone = torch.hub.load(str(REPO_DIR), 'dinov3_convnext_tiny', source='local', pretrained=False)
state_dict = torch.load(LOCAL_WEIGHTS, map_location=DEVICE)
backbone.load_state_dict(state_dict)
backbone = backbone.to(DEVICE)
backbone.eval()

dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
with torch.no_grad():
    raw_feats = backbone(dummy_input)
    if raw_feats.dim() == 2:
        pooled_feats = raw_feats
    elif raw_feats.dim() == 4:
        pooled_feats = F.adaptive_avg_pool2d(raw_feats, 1).flatten(1)
    else:
        raise ValueError(f"Unexpected raw_feats dim: {raw_feats.dim()}")
HIDDEN_SIZE = pooled_feats.shape[1]

class DinoV3Classifier(nn.Module):
    def __init__(self, backbone, num_classes, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            print("Backbone frozen.")
        self.classifier = nn.Linear(HIDDEN_SIZE, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        if feats.dim() == 2:
            pooled = feats
        elif feats.dim() == 4:
            pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        else:
            raise ValueError(f"Unexpected feats dim: {feats.dim()}")
        return self.classifier(pooled)

model = DinoV3Classifier(backbone, NUM_CLASSES).to(DEVICE)

# Load checkpoint
if CHECKPOINT_PATH.exists():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}.")
else:
    raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_PATH}")

# Evaluation function
def evaluate(model, loader, classes):
    model.eval()
    all_preds = []
    all_labels = []
    correct = total = 0
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating Test Set")
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            progress_bar.set_postfix(acc=100 * correct / total)

    # Overall accuracy
    acc = 100 * correct / total
    print(f"\nTest Accuracy: {acc:.2f}%")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for idx, class_name in enumerate(classes):
        print(f"{class_name}: {class_accuracies[idx] * 100:.2f}%")

    return acc

# Run evaluation
test_acc = evaluate(model, test_loader, classes)