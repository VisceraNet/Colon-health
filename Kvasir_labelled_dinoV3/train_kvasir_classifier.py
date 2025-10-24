#!/usr/bin/env python3
# train_kvasir_classifier.py (with patience + graceful stop)
"""
Usage example (recommended inside tmux):
python3 train_kvasir_classifier.py \
  --data_root /home/phil/kvasir_prepared/split \
  --out_dir /mnt/d/kvasir_cls_out \
  --backbone convnext_tiny \
  --image_size 224 \
  --batch_size 96 \
  --epochs 20 \
  --workers 12 \
  --prefetch_factor 8 \
  --amp \
  --no_torch_compile \
  --patience 5
"""

import os, math, time, argparse, random, copy, signal, sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

try:
    import timm
except Exception:
    raise ImportError("Please install timm: pip install timm")

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser(description="Train supervised classifier (patience + graceful stop).")
parser.add_argument("--data_root", required=True)
parser.add_argument("--out_dir", default="./kvasir_cls_out")
parser.add_argument("--backbone", default="convnext_tiny")
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--accumulate", type=int, default=1)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8)//2))
parser.add_argument("--prefetch_factor", type=int, default=4)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--dinov3_repo", type=str, default=None)
parser.add_argument("--dinov3_weights", type=str, default=None)
parser.add_argument("--unfreeze_top_k", type=int, default=0)
parser.add_argument("--save_every", type=int, default=1)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--no_torch_compile", action="store_true")
parser.add_argument("--amp", action="store_true")
parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without val_acc improvement)")
parser.add_argument("--min_delta", type=float, default=0.0, help="Minimum relative improvement (absolute) to count")
args = parser.parse_args()

# -------------------------
# TF32
# -------------------------
def enable_tf32():
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    except Exception:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
enable_tf32()

# -------------------------
# helpers (same as before, robust extractor and prefetcher)
# -------------------------
def extract_tensor(obj):
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, (list, tuple)):
        t = None
        for v in obj:
            if isinstance(v, torch.Tensor):
                t = v; break
            try:
                t = extract_tensor(v); break
            except Exception:
                pass
        if t is None: raise RuntimeError("No tensor found in model output (list/tuple).")
    elif isinstance(obj, dict):
        for v in obj.values():
            try: return extract_tensor(v)
            except Exception: pass
        raise RuntimeError("No tensor found in model output (dict).")
    else:
        raise RuntimeError("Unknown output type from backbone.")
    if t.ndim == 4:
        return t.mean(dim=(2,3))
    if t.ndim == 3:
        # treat as (B,N,C) => mean tokens
        if t.shape[1] > t.shape[2]:
            return t.mean(dim=1)
        else:
            return t.mean(dim=2)
    if t.ndim == 2:
        return t
    return t.flatten(1)

class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.done = False
        self._preload()
    def _preload(self):
        try:
            imgs, targets = next(self.loader)
        except StopIteration:
            self.done = True
            self.next_imgs = None
            self.next_targets = None
            return
        with torch.cuda.stream(self.stream):
            self.next_imgs = imgs.to(self.device, non_blocking=True)
            self.next_targets = targets.to(self.device, non_blocking=True)
    def next(self):
        if self.done: return None
        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        imgs, targets = self.next_imgs, self.next_targets
        self._preload()
        return imgs, targets

# -------------------------
# backbone/head builder (timm fallback)
# -------------------------
def build_backbone_and_head(device, num_classes):
    if args.dinov3_repo:
        try:
            import sys, importlib
            repo_path = Path(args.dinov3_repo).resolve()
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
            dinov3 = importlib.import_module("dinov3")
            # find a callable
            cand = [n for n in dir(dinov3) if n.startswith("dinov3")]
            if cand:
                model = getattr(dinov3, cand[0])(pretrained=False)
                model.to(device)
                print("Using local dinov3:", cand[0])
                def wrapped(x):
                    out = model(x)
                    return extract_tensor(out)
                with torch.no_grad():
                    feat = wrapped(torch.randn(1,3,args.image_size,args.image_size).to(device))
                feat_dim = feat.shape[1]
                head = nn.Linear(feat_dim, num_classes)
                return wrapped, head, feat_dim
        except Exception as e:
            print("dinov3 local failed:", e)
    # fallback timm
    net = timm.create_model(args.backbone, pretrained=True, num_classes=0, global_pool="")
    net.to(device)
    class Wrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, x):
            out = self.net.forward_features(x) if hasattr(self.net, "forward_features") else self.net(x)
            return extract_tensor(out)
    wrapped = Wrapper(net).to(device)
    with torch.no_grad():
        feat = wrapped(torch.randn(1,3,args.image_size,args.image_size).to(device))
        feat_dim = feat.shape[1]
    head = nn.Linear(feat_dim, num_classes)
    return wrapped, head, feat_dim

def freeze_backbone(mod, unfreeze_top_k=0):
    for p in mod.parameters(): p.requires_grad = False
    raw = getattr(mod, "net", getattr(mod, "model", mod))
    if unfreeze_top_k > 0:
        if hasattr(raw, "blocks"):
            blocks = list(raw.blocks)
            n = len(blocks)
            start = max(0, n - unfreeze_top_k)
            for i in range(start, n):
                for p in blocks[i].parameters(): p.requires_grad = True
            if hasattr(raw, "norm"):
                for p in raw.norm.parameters(): p.requires_grad = True
        else:
            params = list(raw.parameters())
            for p in params[-unfreeze_top_k:]:
                p.requires_grad = True

# -------------------------
# signal handling: save checkpoint on interrupt and exit
# -------------------------
interrupt_triggered = False
def _save_checkpoint_on_signal(backbone, head, optimizer, epoch, out_dir):
    out_dir = Path(out_dir)
    ck = {
        "epoch": epoch,
        "backbone": getattr(backbone, "state_dict", lambda: {})(),
        "classifier": getattr(head, "state_dict", lambda: {})(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "note": "saved_on_signal"
    }
    p = out_dir / "ckpt_interrupt.pth"
    torch.save(ck, p)
    print(f"\nSaved interrupt checkpoint to {p}")

def _signal_handler(sig, frame):
    global interrupt_triggered
    print(f"\nReceived signal {sig}. Will attempt graceful shutdown after current minibatch...")
    interrupt_triggered = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# -------------------------
# train / eval loops
# -------------------------
def train_epoch(backbone, head, loader, optimizer, scaler, device, epoch, accumulate_steps, amp):
    backbone.eval()
    head.train()
    running_loss = 0.0; running_correct = 0; running_n = 0
    prefetch = DataPrefetcher(loader, device)
    it = 0
    optimizer.zero_grad()
    pbar = tqdm(total=len(loader), desc=f"Train E{epoch}")
    global interrupt_triggered
    while True:
        if interrupt_triggered:
            print("Interrupt flagged — finishing current loop and exiting epoch.")
        batch = prefetch.next()
        if batch is None: break
        imgs, targets = batch
        with torch.no_grad():
            if amp:
                with torch.cuda.amp.autocast(enabled=True):
                    feats = backbone(imgs)
            else:
                feats = backbone(imgs)
        if amp:
            with torch.cuda.amp.autocast(enabled=True):
                outputs = head(feats)
                loss = F.cross_entropy(outputs, targets)
            scaler.scale(loss / accumulate_steps).backward()
            if (it + 1) % accumulate_steps == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        else:
            outputs = head(feats)
            loss = F.cross_entropy(outputs, targets)
            loss = loss / accumulate_steps
            loss.backward()
            if (it + 1) % accumulate_steps == 0:
                optimizer.step(); optimizer.zero_grad()
        preds = outputs.argmax(dim=1)
        running_correct += (preds == targets).sum().item()
        running_n += imgs.size(0)
        running_loss += float(loss.item()) * imgs.size(0) * (1 if amp else accumulate_steps)
        it += 1
        pbar.update(1)
        pbar.set_postfix({"loss": running_loss / running_n, "acc": running_correct / running_n})
        if interrupt_triggered:
            break
    pbar.close()
    return running_loss / max(1, running_n), running_correct / max(1, running_n)

@torch.no_grad()
def evaluate(backbone, head, loader, device, amp):
    backbone.eval(); head.eval()
    running_loss = 0.0; running_correct = 0; running_n = 0
    prefetch = DataPrefetcher(loader, device)
    while True:
        batch = prefetch.next()
        if batch is None: break
        imgs, targets = batch
        if amp:
            with torch.cuda.amp.autocast(enabled=True):
                feats = backbone(imgs)
                outputs = head(feats)
                loss = F.cross_entropy(outputs, targets, reduction="sum")
        else:
            feats = backbone(imgs)
            outputs = head(feats)
            loss = F.cross_entropy(outputs, targets, reduction="sum")
        running_loss += float(loss.item())
        running_correct += (outputs.argmax(dim=1) == targets).sum().item()
        running_n += imgs.size(0)
    return running_loss / max(1, running_n), running_correct / max(1, running_n)

# -------------------------
# main
# -------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    data_root = Path(args.data_root)
    train_dir = data_root / "train"; val_dir = data_root / "val"; test_dir = data_root / "test"
    if not train_dir.exists():
        raise SystemExit("train dir missing under data_root")
    # transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.8,1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(args.image_size*1.14), interpolation=Image.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=val_tf)
    num_classes = len(train_ds.classes)
    print("Classes:", num_classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, persistent_workers=True,
                              prefetch_factor=args.prefetch_factor, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=True,
                            prefetch_factor=args.prefetch_factor)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, persistent_workers=True,
                             prefetch_factor=args.prefetch_factor)

    backbone, head, feat_dim = build_backbone_and_head(device, num_classes)
    head.to(device)
    freeze_backbone(backbone, unfreeze_top_k=args.unfreeze_top_k)
    for p in head.parameters(): p.requires_grad = True

    params = [p for p in list(head.parameters()) + list(backbone.parameters()) if p.requires_grad]
    eff_batch = args.batch_size * max(1, args.accumulate)
    base_lr = args.lr if args.lr is not None else 1e-3
    scaled_lr = base_lr * eff_batch / 256.0
    optimizer = torch.optim.SGD(params, lr=scaled_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print(f"LR={scaled_lr:.6g} eff_batch={eff_batch}, trainable_params={sum(p.numel() for p in params)}")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if not args.no_torch_compile:
        try:
            if hasattr(torch, "compile"):
                backbone = torch.compile(backbone)
                head = torch.compile(head)
                print("Applied torch.compile (if supported).")
        except Exception as e:
            print("torch.compile failed:", e)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0; best_epoch = 0
    epochs_no_improve = 0
    start_epoch = 1

    # resume
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        if "classifier" in ck:
            head.load_state_dict(ck["classifier"], strict=False)
        if "backbone" in ck:
            try: backbone.load_state_dict(ck["backbone"], strict=False)
            except Exception: pass
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck.get("epoch", 0) + 1
        best_val = ck.get("best_val", best_val)
        print("Resumed from", args.resume, "start_epoch", start_epoch, "best_val", best_val)

    global interrupt_triggered
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc = train_epoch(backbone, head, train_loader, optimizer, scaler, device, epoch, args.accumulate, args.amp)
        scheduler.step()
        val_loss, val_acc = evaluate(backbone, head, val_loader, device, args.amp)
        print(f"Epoch {epoch} results: train_loss {train_loss:.4f} train_acc {train_acc:.4f} val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

        # save checkpoint
        ck = {
            "epoch": epoch,
            "backbone": getattr(backbone, "state_dict", lambda: {})(),
            "classifier": getattr(head, "state_dict", lambda: {})(),
            "optimizer": optimizer.state_dict(),
            "best_val": best_val
        }
        torch.save(ck, out_dir / f"ckpt_epoch_{epoch}.pth")
        if val_acc > best_val + args.min_delta:
            best_val = val_acc
            best_epoch = epoch
            torch.save(ck, out_dir / "best.pth")
            print("New best saved (val_acc improved).")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement this epoch ({epochs_no_improve}/{args.patience})")

        # early stopping check
        if epochs_no_improve >= args.patience:
            print(f"Stopping early (patience {args.patience} reached). Best epoch: {best_epoch}, best_val {best_val}")
            break

        # if signal triggered during epoch, break and save
        if interrupt_triggered:
            print("Interrupt detected: saving interrupt checkpoint and exiting.")
            _save_checkpoint_on_signal(backbone, head, optimizer, epoch, out_dir)
            break

    # final test evaluation on best or last checkpoint
    # load best if exists
    best_path = out_dir / "best.pth"
    if best_path.exists():
        ck = torch.load(best_path, map_location="cpu")
        try:
            head.load_state_dict(ck.get("classifier", {}), strict=False)
            backbone.load_state_dict(ck.get("backbone", {}), strict=False)
        except Exception:
            pass
        print("Loaded best checkpoint for final test.")
    test_loss, test_acc = evaluate(backbone, head, test_loader, device, args.amp)
    print(f"Final test: loss {test_loss:.4f}, acc {test_acc:.4f}")
    print("Done. Checkpoints in:", out_dir)

if __name__ == "__main__":
    main()
