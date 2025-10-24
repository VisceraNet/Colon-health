#!/usr/bin/env python3
# eval_kvasir_best.py
# Usage examples:
# 1) Labeled folder (ImageFolder style test split):
# python3 eval_kvasir_best.py --test_root /home/phil/kvasir_prepared/split/test --ckpt /mnt/d/kvasir_cls_out/best.pth --out_csv /mnt/d/kvasir_preds.csv --batch_size 128 --workers 8
#
# 2) Unlabeled folder of images:
# python3 eval_kvasir_best.py --test_root /home/phil/unseen_images --ckpt /mnt/d/kvasir_cls_out/best.pth --out_csv /mnt/d/kvasir_preds_unlabeled.csv --batch_size 128 --workers 8
#
# 3) If you trained with local dinov3 repo and want to use it to build the same backbone:
# python3 eval_kvasir_best.py --test_root ... --ckpt ... --dinov3_repo /home/phil/dinov3 --dinov3_weights /home/phil/dinov3_weights.pth

import argparse
import os
from pathlib import Path
import csv
import time
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image

# timm fallback
try:
    import timm
except Exception:
    timm = None

# ------------------------------
# CLI
# ------------------------------
parser = argparse.ArgumentParser(description="Evaluate classifier on test images using best.pth")
parser.add_argument("--test_root", required=True, help="Folder with test images. Either ImageFolder style (subfolders per class) or flat folder of images.")
parser.add_argument("--ckpt", required=True, help="Path to checkpoint (best.pth) produced by training script.")
parser.add_argument("--out_csv", default="predictions.csv", help="CSV path to save predictions.")
parser.add_argument("--backbone", default="convnext_tiny", help="timm backbone name fallback.")
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--amp", action="store_true", help="Use mixed precision for inference.")
parser.add_argument("--dinov3_repo", type=str, default=None, help="Optional local dinov3 repo path.")
parser.add_argument("--dinov3_weights", type=str, default=None, help="Optional dinov3 weights path.")
args = parser.parse_args()

# ------------------------------
# Helpers
# ------------------------------
def extract_tensor(obj):
    """Robustly extract a (B, C) tensor from model output."""
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            if isinstance(v, torch.Tensor):
                t = v; break
            try:
                t = extract_tensor(v); break
            except Exception:
                pass
        else:
            raise RuntimeError("No tensor found in list/tuple output.")
    elif isinstance(obj, dict):
        for v in obj.values():
            try:
                return extract_tensor(v)
            except Exception:
                pass
        raise RuntimeError("No tensor found in dict output.")
    else:
        raise RuntimeError("Unsupported model output type.")

    # handle shapes
    if t.ndim == 4:
        return t.mean(dim=(2,3))
    if t.ndim == 3:
        # assume (B, N, C)
        if t.shape[1] > t.shape[2]:
            return t.mean(dim=1)
        else:
            return t.mean(dim=2)
    if t.ndim == 2:
        return t
    return t.flatten(1)

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

# Data loader for unlabeled flat folder
class FlatFolderDataset(Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.files = sorted([p for p in self.root.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".tiff",".webp")])
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), str(p)

# DataPrefetcher to overlap host->device transfers
class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.done = False
        self._preload()
    def _preload(self):
        try:
            batch = next(self.loader)
        except StopIteration:
            self.done = True
            self.next_batch = None
            return
        imgs, rest = batch
        with torch.cuda.stream(self.stream):
            self.next_imgs = imgs.to(self.device, non_blocking=True)
            self.next_rest = rest
    def next(self):
        if self.done:
            return None
        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        imgs = self.next_imgs
        rest = self.next_rest
        self._preload()
        return imgs, rest

# ------------------------------
# Build backbone (dinov3 local preferred, else timm)
# ------------------------------
def try_load_dinov3(repo_dir, model_name="dinov3_vits14", weights=None, device="cuda"):
    """
    Try to import local dinov3 repo and create model.
    Returns module or raises.
    """
    try:
        import importlib, sys
        repo = Path(repo_dir).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        dinov3 = importlib.import_module("dinov3")
        if hasattr(dinov3, model_name):
            model = getattr(dinov3, model_name)(pretrained=False)
            model.to(device)
            if weights:
                sd = torch.load(weights, map_location="cpu")
                try:
                    model.load_state_dict(sd, strict=False)
                except Exception:
                    model.load_state_dict(sd.get("state_dict", sd), strict=False)
            return model
        else:
            raise RuntimeError("Model name not found in dinov3 package.")
    except Exception as e:
        raise RuntimeError(f"dinov3 import/load failed: {e}")

def build_backbone_wrapper(device):
    # try dinov3 if requested
    if args.dinov3_repo:
        try:
            print("Attempting to load dinov3 from:", args.dinov3_repo)
            model = try_load_dinov3(args.dinov3_repo, weights=args.dinov3_weights, device=device)
            def forward_fn(x):
                return extract_tensor(model(x))
            return forward_fn
        except Exception as e:
            print("dinov3 load failed:", e, " -> falling back to timm")

    # fallback to timm
    if timm is None:
        raise RuntimeError("timm not available; cannot build backbone.")
    print("Using timm backbone:", args.backbone)
    net = timm.create_model(args.backbone, pretrained=True, num_classes=0, global_pool="")
    net.to(device)
    def forward_fn(x):
        out = net.forward_features(x) if hasattr(net, "forward_features") else net(x)
        return extract_tensor(out)
    return forward_fn

# ------------------------------
# Load checkpoint and build head
# ------------------------------
def load_checkpoint_and_build(model_forward, ckpt_path, device, class_names=None):
    """
    ckpt expected to contain at least 'classifier' or 'head' or similar.
    Returns (head_module, class_names, feat_dim)
    """
    ck = torch.load(ckpt_path, map_location="cpu")
    # find classifier weights in checkpoint
    candidate_keys = []
    for k in ck.keys():
        candidate_keys.append(k)
    # possible keys: 'classifier', 'head', 'model', 'state_dict', 'student_state', etc.
    # try common patterns:
    state_dict = None
    if "classifier" in ck:
        state_dict = ck["classifier"]
    elif "head" in ck:
        state_dict = ck["head"]
    elif "model" in ck:
        state_dict = ck["model"]
    elif "state_dict" in ck:
        # flat state dict: find linear weight shapes
        sd = ck["state_dict"]
        # try to find last linear layer keys
        for name, val in sd.items():
            if val.ndim == 2:
                # candidate head weight
                state_dict = {}
                # extract keys that look like head
                for k,v in sd.items():
                    if k.startswith(name.split('.')[0]):
                        state_dict[k] = v
                break
        if state_dict is None:
            state_dict = sd
    else:
        # maybe whole checkpoint is a state dict for model + head
        if isinstance(ck, dict) and all(isinstance(v, torch.Tensor) for v in ck.values()):
            state_dict = ck
        else:
            # try nested 'student_state' etc.
            if "student_state" in ck and isinstance(ck["student_state"], dict):
                state_dict = ck["student_state"]
            elif "teacher_state" in ck and isinstance(ck["teacher_state"], dict):
                state_dict = ck["teacher_state"]
    # If still None, try to look for any tensor 2D shapes in ck
    if state_dict is None:
        for k, v in ck.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                # single matrix - treat ck as direct head state (rare)
                state_dict = {k: v}
                break

    # if class_names provided (ImageFolder), use that to construct head; otherwise infer from state_dict shapes
    # we need feat_dim: run a dummy input through model_forward
    dummy = torch.randn(1,3,args.image_size,args.image_size).to(device)
    with torch.no_grad():
        feat = model_forward(dummy)
    feat = feat.to("cpu")
    if feat.ndim != 2:
        feat = feat.flatten(1)
    feat_dim = feat.shape[1]

    if class_names is not None:
        num_classes = len(class_names)
    else:
        # infer num classes from state_dict shapes if possible
        num_classes = None
        if isinstance(state_dict, dict):
            # find a weight matrix whose second dim equals feat_dim
            for k,v in state_dict.items():
                if isinstance(v, torch.Tensor) and v.ndim == 2:
                    # head weight shape is (num_classes, feat_dim) or (feat_dim, num_classes)
                    if v.shape[1] == feat_dim:
                        num_classes = v.shape[0]
                        break
                    if v.shape[0] == feat_dim:
                        num_classes = v.shape[1]
                        break
        if num_classes is None:
            raise RuntimeError("Cannot infer number of classes from checkpoint. Provide a labeled test folder or pass class_names.")

    # build a linear head
    head = nn.Linear(feat_dim, num_classes)
    head.to(device)

    # now attempt to load state into head (best-effort)
    try:
        # if state_dict already matches head.state_dict keys, load directly
        if isinstance(state_dict, dict) and set(state_dict.keys()) >= set(head.state_dict().keys()):
            # convert tensors to match device
            sd = {k: torch.as_tensor(v) for k,v in state_dict.items()}
            head.load_state_dict(sd, strict=False)
        else:
            # try to find weight/bias matrices in state_dict and assign
            w = None; b = None
            for v in state_dict.values():
                if isinstance(v, torch.Tensor) and v.ndim == 2:
                    vv = v
                    if vv.shape[1] == feat_dim:
                        # shape (num_classes, feat_dim)
                        w = vv
                        break
                    if vv.shape[0] == feat_dim:
                        # shape (feat_dim, num_classes) -> transpose
                        w = vv.t()
                        break
            # biases
            for v in state_dict.values():
                if isinstance(v, torch.Tensor) and v.ndim == 1:
                    if w is not None and v.shape[0] == w.shape[0]:
                        b = v
                        break
            if w is not None:
                w = torch.as_tensor(w)
                if w.device != head.weight.device:
                    w = w.to(head.weight.device)
                head.weight.data.copy_(w)
            if b is not None:
                b = torch.as_tensor(b)
                if b.device != head.bias.device:
                    b = b.to(head.bias.device)
                head.bias.data.copy_(b)
    except Exception as e:
        print("Warning: could not fully load head from checkpoint; proceeding with partial/strict=False. Error:", e)
        try:
            head.load_state_dict(state_dict, strict=False)
        except Exception:
            pass

    # build class_names mapping if not supplied
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    return head, class_names, feat_dim

# ------------------------------
# Main evaluation routine
# ------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    test_root = Path(args.test_root)
    if not test_root.exists():
        raise SystemExit("test_root does not exist.")

    # choose dataset type
    # if test_root contains subdirs, treat as ImageFolder
    subdirs = [p for p in test_root.iterdir() if p.is_dir()]
    is_imagefolder = len(subdirs) > 0
    transform = transforms.Compose([
        transforms.Resize(int(args.image_size * 1.14), interpolation=Image.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    if is_imagefolder:
        dataset = datasets.ImageFolder(test_root, transform=transform)
        class_names = dataset.classes
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=True)
        labeled = True
        print(f"Detected labeled ImageFolder with {len(dataset)} images and {len(class_names)} classes.")
    else:
        dataset = FlatFolderDataset(test_root, transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=True)
        class_names = None
        labeled = False
        print(f"Detected unlabeled flat folder with {len(dataset)} images.")

    # build backbone forward fn
    forward_fn = build_backbone_wrapper(device)

    # build head from checkpoint
    head, class_names, feat_dim = load_checkpoint_and_build(forward_fn, args.ckpt, device, class_names=class_names)
    head.to(device)
    head.eval()

    # move forward_fn to device: forward_fn may be a function using a model already on device.
    # We'll call forward_fn inside torch.no_grad().

    # inference
    total = 0
    correct = 0
    rows = []
    prefetch = DataPrefetcher(loader, device)
    t0 = time.time()
    it = 0
    pbar = tqdm(total=len(loader), desc="Eval")
    while True:
        batch = prefetch.next()
        if batch is None:
            break
        imgs, rest = batch  # rest: either targets (if ImageFolder) or file paths (if FlatFolderDataset)
        with torch.no_grad():
            if args.amp:
                with torch.cuda.amp.autocast(enabled=True):
                    feats = forward_fn(imgs)
                    outputs = head(feats)
            else:
                feats = forward_fn(imgs)
                outputs = head(feats)
            probs = F.softmax(outputs, dim=-1)
            topv, topi = probs.max(dim=1)
        # handle rest
        if labeled:
            targets = rest
            # rest is a tensor of targets on CPU? ImageFolder yields tensors -> when moved we passed only images to device, but DataPrefetcher stored only imgs->device and next_rest kept original "targets" (not moved). To ensure correctness, we handle rest as targets on CPU (torch.Tensor)
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy().tolist()
            for j in range(len(topi)):
                pred_idx = int(topi[j].item())
                pred_label = class_names[pred_idx] if class_names else str(pred_idx)
                prob = float(topv[j].item())
                true_idx = int(targets[j])
                true_label = class_names[true_idx] if class_names else str(true_idx)
                rows.append({"path": "", "pred_idx": pred_idx, "pred_label": pred_label, "prob": prob, "true_idx": true_idx, "true_label": true_label})
                if pred_idx == true_idx:
                    correct += 1
                total += 1
        else:
            # rest contains paths (strings)
            for j in range(len(topi)):
                pred_idx = int(topi[j].item())
                pred_label = class_names[pred_idx] if class_names else str(pred_idx)
                prob = float(topv[j].item())
                path = rest[j] if isinstance(rest, (list,tuple)) else rest[j].item() if hasattr(rest[j], "item") else str(rest[j])
                rows.append({"path": path, "pred_idx": pred_idx, "pred_label": pred_label, "prob": prob})
                total += 1

        it += 1
        pbar.update(1)
    pbar.close()
    dt = time.time() - t0

    # save CSV
    out_path = Path(args.out_csv)
    fieldnames = None
    if labeled:
        fieldnames = ["path","pred_idx","pred_label","prob","true_idx","true_label"]
    else:
        fieldnames = ["path","pred_idx","pred_label","prob"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # ensure path exists in labeled mode: if ImageFolder, we didn't store paths earlier; reconstruct them:
            if labeled and r["path"] == "":
                # we didn't record path to avoid extra copies; recover by iterating dataset again (cheap)
                pass
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # If labeled, compute accuracy
    if labeled and total > 0:
        acc = correct / total
        print(f"Accuracy: {acc*100:.3f}% ({correct}/{total}) - time {dt:.1f}s")
    else:
        print(f"Done. Predicted {total} images. Results saved to {out_path}. Time {dt:.1f}s")

if __name__ == "__main__":
    main()
