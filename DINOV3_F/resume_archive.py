import json
import math
import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

# Config
DATA_ROOT = Path("/mnt/d/Project/archive")
NUM_CLASSES = 4
REPO_DIR = Path("/home/phil/dinov3")
LOCAL_WEIGHTS = "/mnt/d/Project/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE = 5
MIN_DELTA = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
CKPT_DIR = Path("dinov3_archive_checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = CKPT_DIR / "best_model.pt"

print(f"Using device: {DEVICE}")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_split(split_name, transform):
    split_path = DATA_ROOT / split_name
    dataset = datasets.ImageFolder(root=split_path, transform=transform)
    print(f"{split_name.capitalize()} dataset: {len(dataset)} samples")
    print(f"{split_name.capitalize()} class distribution: {Counter(dataset.targets)}")
    return dataset

train_dataset = load_split("train", train_transform)
val_dataset = load_split("val", val_test_transform)
test_dataset = load_split("test", val_test_transform)

classes = train_dataset.classes
print(f"Classes: {classes}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
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
backbone_config = {"hidden_size": HIDDEN_SIZE}

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

# Training setup
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = EPOCHS * len(train_loader)
warmup_steps = int(WARMUP_RATIO * total_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
criterion = nn.CrossEntropyLoss()

# Evaluation function
def evaluate(model, loader, criterion=None):
    model.eval()
    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating")
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            if criterion:
                loss = criterion(logits, labels)
                total_loss += loss.item()
    acc = 100 * correct / total
    if criterion:
        avg_loss = total_loss / len(loader)
        print(f"Eval Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
    return acc

# Load checkpoint
start_epoch = 0
best_val_acc = 0.0
if CHECKPOINT_PATH.exists():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = evaluate(model, val_loader, criterion)  # Get current best val acc
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}. Resuming from epoch {start_epoch}")
else:
    print("No checkpoint found, starting from scratch.")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            return False
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

for epoch in range(start_epoch, EPOCHS):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch in progress_bar:
        images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
    
    val_acc = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch+1} Val Acc: {val_acc:.4f}")
    
    if early_stopping(val_acc):
        print(f"Early stopping at epoch {epoch+1}. Best Val Acc: {best_val_acc:.4f}")
        break
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {
                "classes": classes,
                "backbone_config": backbone_config,
                "num_classes": NUM_CLASSES,
            },
            "epoch": epoch,
        }, CHECKPOINT_PATH)
        print(f"New best model saved! Acc: {val_acc:.4f}")

test_acc = evaluate(model, test_loader)
print(f"Final Test Accuracy: {test_acc:.2f}%")

def predict_image(model, image_path, classes, transform, device):
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logit = model(image_tensor)
        prob = torch.softmax(logit, dim=1)
        pred_class = classes[prob.argmax().item()]
        conf = prob.max().item()
    return pred_class, conf