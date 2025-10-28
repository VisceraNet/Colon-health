# train_severity.py
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from PIL import Image

# Config
DATA_ROOT = Path("/home/phil/LMIC/split")
NUM_CLASSES = 4
REPO_DIR = Path("/home/phil/dinov3")
LOCAL_WEIGHTS = "/mnt/d/Project/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
BATCH_SIZE = 64
EPOCHS = 100
LR = 5e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE = 20
MIN_DELTA = 0.001
RESUME = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
CKPT_DIR = Path("dinov3_lmic_checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

print(f"Using device: {DEVICE}")

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
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

# Compute class weights for handling imbalance in training data
class_counts = Counter(train_dataset.targets)
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[train_dataset.targets[i]] for i in range(len(train_dataset))]
weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Model load
backbone = torch.hub.load(str(REPO_DIR), 'dinov3_vits16', source='local', pretrained=False)
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

class OrdinalClassifier(nn.Module):
    def __init__(self, backbone, num_classes, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            print("Backbone frozen.")
        self.num_classes = num_classes
        self.classifier = nn.Linear(HIDDEN_SIZE, num_classes - 1)

    def forward(self, x):
        feats = self.backbone(x)
        if feats.dim() == 2:
            pooled = feats
        elif feats.dim() == 4:
            pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        else:
            raise ValueError(f"Unexpected feats dim: {feats.dim()}")
        logits = self.classifier(pooled)  # [B, K-1]
        return logits

model = OrdinalClassifier(backbone, NUM_CLASSES).to(DEVICE)

def ordinal_bce_loss(logits, labels):
    # logits: [B, K-1], labels: [B] in 0..K-1
    B = logits.size(0)
    targets = torch.zeros_like(logits)
    for i in range(B):
        targets[i, :labels[i]] = 1.0
    return F.binary_cross_entropy_with_logits(logits, targets)

# Training setup
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = EPOCHS * len(train_loader)
warmup_steps = int(WARMUP_RATIO * total_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

def weighted_precision_recall(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_classes = int(np.max(y_true)) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    precisions = np.diag(cm) / np.sum(cm, axis=0)
    recalls = np.diag(cm) / np.sum(cm, axis=1)
    precisions[np.isnan(precisions)] = 0
    recalls[np.isnan(recalls)] = 0
    supports = np.sum(cm, axis=1)
    total = np.sum(supports)
    if total == 0:
        return 0, 0
    w_precision = np.average(precisions, weights=supports)
    w_recall = np.average(recalls, weights=supports)
    return w_precision, w_recall

def quadratic_weighted_kappa(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n_rater = y_true.shape[0]
    n_classes = int(np.max(np.concatenate((y_true, y_pred)))) + 1
    conf_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_rater):
        conf_matrix[y_true[i], y_pred[i]] += 1
    conf_matrix = conf_matrix / n_rater
    sum_row = np.sum(conf_matrix, axis=1)
    sum_col = np.sum(conf_matrix, axis=0)
    exp_matrix = np.outer(sum_row, sum_col)
    weight_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            weight_matrix[i][j] = ((i - j) / (n_classes - 1.0)) ** 2 if n_classes > 1 else 0
    num = np.sum(conf_matrix * weight_matrix)
    den = np.sum(exp_matrix * weight_matrix)
    if den == 0:
        return 1.0
    kappa = 1.0 - num / den
    return kappa

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
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
best_val_qwk = None

# Resume
start_epoch = 0
ckpt_path = CKPT_DIR / "best_model.pt"
if RESUME and ckpt_path.exists():
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_qwk = checkpoint.get('best_qwk', None)
    early_stopping.best_score = best_val_qwk
    print(f"Resumed from epoch {start_epoch}, best QWK: {best_val_qwk:.4f}")

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating")
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            # Predict: number of thresholds where sigmoid > 0.5
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).sum(dim=1).long()
            # Clamp to 0..K-1
            preds = torch.clamp(preds, 0, model.num_classes - 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loss = ordinal_bce_loss(logits, labels)
            total_loss += loss.item()
    acc = 100 * correct / total
    avg_loss = total_loss / len(loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    precision, recall = weighted_precision_recall(all_labels, all_preds)
    qwk_score = quadratic_weighted_kappa(all_labels, all_preds)
    print(f"Eval Loss: {avg_loss:.4f}, Acc: {acc:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, QWK: {qwk_score:.4f}")
    return {'acc': acc, 'qwk': qwk_score}

for epoch in range(start_epoch, EPOCHS):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch in progress_bar:
        images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = ordinal_bce_loss(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
    
    val_metrics = evaluate(model, val_loader)
    val_qwk = val_metrics['qwk']
    
    if early_stopping(val_qwk):
        print(f"Early stopping at epoch {epoch+1}. Best Val QWK: {early_stopping.best_score:.4f}")
        break
    
    if val_qwk > (best_val_qwk or -np.inf):
        best_val_qwk = val_qwk
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_qwk": best_val_qwk,
            "config": {
                "classes": classes,
                "backbone_config": backbone_config,
                "num_classes": NUM_CLASSES,
            },
            "epoch": epoch,
        }, CKPT_DIR / "best_model.pt")
        print(f"New best model saved! QWK: {val_qwk:.4f}")

test_metrics = evaluate(model, test_loader)

def predict_image(model, image_path, classes, transform, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logit = model(image_tensor)
        probs = torch.sigmoid(logit)
        pred_idx = (probs > 0.5).sum().item()
        pred_idx = min(pred_idx, len(classes) - 1)
        pred_class = classes[pred_idx]
        conf = probs.mean().item()  # Approximate conf
    return pred_class, conf