# train_fusion_classification_f1.py
import os
import random
import csv
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import timm

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score
)

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 100
LR = 2e-4
WEIGHT_DECAY = 1e-4
EFF_WEIGHT = 0.75
PATIENCE = 10
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = r"D:\LIMUC"
CHECKPOINT_PATH = "best_fusion_f1.pth"
PLOTS_DIR = "training_plots_f1"

os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -----------------------------
# Simple ImageFolder Dataset (NO MASK)
# -----------------------------
class SimpleImageFolder(Dataset):
    def __init__(self, root):
        self.samples = []
        self.classes = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for c in self.classes:
            cls_dir = os.path.join(root, c)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[c])
                    )

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

# -----------------------------
# Fusion Model (MULTICLASS)
# -----------------------------
class ResEffFusion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.eff = timm.create_model(
            "efficientnet_b4", pretrained=True, features_only=True
        )
        self.res = timm.create_model(
            "resnet50", pretrained=True, features_only=True
        )

        eff_dim = self.eff.feature_info[-1]['num_chs']
        res_dim = self.res.feature_info[-1]['num_chs']

        self.eff_proj = nn.Conv2d(eff_dim, 1024, 1)
        self.res_proj = nn.Conv2d(res_dim, 1024, 1)

        self.bn = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        eff_feat = self.eff(x)[-1]
        res_feat = self.res(x)[-1]

        if eff_feat.shape[2:] != res_feat.shape[2:]:
            res_feat = nn.functional.interpolate(
                res_feat, size=eff_feat.shape[2:],
                mode="bilinear", align_corners=False
            )

        eff_f = self.eff_proj(eff_feat)
        res_f = self.res_proj(res_feat)

        fused = 0.75 * eff_f + 0.25 * res_f
        fused = self.relu(self.bn(fused))

        pooled = self.pool(fused).flatten(1)
        pooled = self.dropout(pooled)

        return self.classifier(pooled)

# -----------------------------
# EarlyStopping (F1-based)
# -----------------------------
class EarlyStopping:
    def __init__(self, patience):
        self.best = -1
        self.counter = 0
        self.patience = patience

    def step(self, metric):
        if metric > self.best:
            self.best = metric
            self.counter = 0
            return False, True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True, False
            return False, False

# -----------------------------
# Training Pipeline
# -----------------------------
def train():

    train_ds = SimpleImageFolder(os.path.join(DATA_ROOT, "train"))
    val_ds   = SimpleImageFolder(os.path.join(DATA_ROOT, "val"))
    test_ds  = SimpleImageFolder(os.path.join(DATA_ROOT, "test"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    NUM_CLASSES = len(train_ds.classes)
    print("Classes:", train_ds.classes)

    model = ResEffFusion(NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    earlystop = EarlyStopping(PATIENCE)

    for epoch in range(EPOCHS):

        # ---------------- TRAIN ----------------
        model.train()
        running_loss = 0

        for imgs, labels in tqdm(train_loader,
                                 desc=f"Epoch {epoch+1}/{EPOCHS}"):

            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # ---------------- VALIDATION ----------------
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)

                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

        print(f"Val → Acc: {acc:.4f} | F1: {macro_f1:.4f} | QWK: {qwk:.4f}")

        stop, is_best = earlystop.step(macro_f1)

        if is_best:
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("Saved best model (F1 improved)")

        if stop:
            print("Early stopping.")
            break

    # ---------------- TEST ----------------
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1 (PRIMARY): {macro_f1:.4f}")
    print(f"QWK (secondary): {qwk:.4f}")
    print(classification_report(y_true, y_pred,
                                target_names=train_ds.classes))

if __name__ == "__main__":
    train()