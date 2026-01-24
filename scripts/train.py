# train.py

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from datasets import LIMUCDataset
from models import EFFResNetViT
from losses import binary_loss_fn, listnet_loss
from evaluate import evaluate
from config import *
import os
from pathlib import Path
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import random
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer



# -------------------------
# helpers
# -------------------------
def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics
    }, path)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility (slower but deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# training
# -------------------------
def train():
    set_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data/train_and_validation_sets"

    train_ds = LIMUCDataset(str(DATA_DIR), train=True)
    val_ds   = LIMUCDataset(str(DATA_DIR), train=False)

    labels = [s[1] for s in train_ds.samples]
    class_counts = Counter(labels)

    # Inverse frequency
    weights = [1.0 / class_counts[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=SAMPLES_PER_EPOCH,
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False
    )

    model = EFFResNetViT().to(DEVICE)

    # ---- freeze CNNs + transformer warmup
    set_requires_grad(model.eff, False)
    set_requires_grad(model.res, False)
    set_requires_grad(model.transformer, False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scaler = GradScaler()

    best_recall = 0.0
    patience = 5
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        # ---- unfreeze after warmup
        if epoch == FREEZE_BACKBONES_EPOCHS:
            set_requires_grad(model, True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=UNFREEZE_LR,
                weight_decay=WEIGHT_DECAY
            )

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = batch["image"].to(DEVICE)
            binary = batch["binary"].to(DEVICE)
            ordinal = batch["ordinal"].to(DEVICE)

            optimizer.zero_grad()

            # 🔥 AMP STARTS HERE
            with autocast(device_type="cuda"):
                out = model(imgs)

                loss_bin = binary_loss_fn(out["binary_logits"], binary)

                if ordinal.unique().numel() > 1:
                    loss_rank = listnet_loss(out["severity_score"], ordinal)
                else:
                    loss_rank = torch.tensor(0.0, device=DEVICE)

                loss = (
                    LAMBDA_BINARY * loss_bin +
                    LAMBDA_RANK * loss_rank
                )
            # 🔥 AMP ENDS HERE

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        model.eval()

        torch.cuda.synchronize()      # flush queued GPU work
        t0 = timer()

        metrics = evaluate(model, val_loader, DEVICE)

        torch.cuda.synchronize()      # wait for eval kernels
        t1 = timer()

        tqdm.write(f"Eval time: {t1 - t0:.2f}s")

        tqdm.write(
    f"Epoch {epoch+1} | "
    f"Loss: {total_loss/len(train_loader):.4f} | "
    f"Recall: {metrics['recall_active']:.3f} | "
    f"FNR: {metrics['fnr']:.3f} | "
    f"Precision: {metrics['precision']:.3f} | "
    f"QWK: {metrics['qwk']:.3f} | "
    f"Accuracy: {metrics['accuracy']:.3f}"
)
        # ---- early stopping on recall
        if metrics["recall_active"] > best_recall:
            best_recall = metrics["recall_active"]
            no_improve = 0

            save_checkpoint(
                model,
                optimizer,
                epoch,
                metrics,
                f"{CKPT_DIR}/best_model.pt"
            )
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                metrics,
                f"{CKPT_DIR}/epoch_{epoch+1}.pt"
            )

        if no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break


if __name__ == "__main__":
    train()
