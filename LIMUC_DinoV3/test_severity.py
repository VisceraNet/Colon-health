# test_severity.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path
from collections import Counter

# sklearn metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Config
DATA_ROOT = Path("/home/phil/LMIC/split")
CKPT_DIR = Path("dinov3_lmic_checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset & loader
test_dataset = datasets.ImageFolder(root=DATA_ROOT / "test", transform=val_test_transform)
classes = test_dataset.classes
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

print(f"Test dataset: {len(test_dataset)} samples")
print(f"Classes: {classes}")

# Load backbone (local hub repo)
REPO_DIR = Path("/home/phil/dinov3")
LOCAL_WEIGHTS = "/mnt/d/Project/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# Load backbone from local repo via torch.hub (adjust name if needed)
backbone = torch.hub.load(str(REPO_DIR), 'dinov3_vits16', source='local', pretrained=False)

# Load pretrained weights for backbone if available
state_dict = torch.load(LOCAL_WEIGHTS, map_location=DEVICE)
backbone.load_state_dict(state_dict)
backbone = backbone.to(DEVICE)
backbone.eval()

# Determine hidden size from backbone output
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
with torch.no_grad():
    raw_feats = backbone(dummy_input)
    if raw_feats.dim() == 2:
        HIDDEN_SIZE = raw_feats.shape[1]
    else:
        pooled = F.adaptive_avg_pool2d(raw_feats, 1).flatten(1)
        HIDDEN_SIZE = pooled.shape[1]

class OrdinalClassifier(nn.Module):
    def __init__(self, backbone, num_classes, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        # For ordinal classification with k classes we output k-1 logits
        self.classifier = nn.Linear(hidden_size, num_classes - 1)

    def forward(self, x):
        feats = self.backbone(x)
        if feats.dim() == 2:
            pooled = feats
        elif feats.dim() == 4:
            pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        else:
            raise ValueError(f"Unexpected feats dim: {feats.dim()}")
        logits = self.classifier(pooled)
        return logits

NUM_CLASSES = len(classes)
model = OrdinalClassifier(backbone, NUM_CLASSES, HIDDEN_SIZE).to(DEVICE)

# Load checkpoint
ckpt_path = CKPT_DIR / "best_model.pt"
checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}, best QWK: {checkpoint.get('best_qwk', 'N/A')}")

def ordinal_bce_loss(logits, labels):
    """
    logits: (B, K-1)
    labels: (B,) integer class labels in [0, K-1]
    This creates target indicators where for a label y, targets[:y] = 1, rest 0.
    """
    B = logits.size(0)
    targets = torch.zeros_like(logits, dtype=logits.dtype, device=logits.device)
    # For each sample set the first labels[i] positions to 1
    for i in range(B):
        y = int(labels[i].item()) if isinstance(labels[i], torch.Tensor) else int(labels[i])
        if y > 0:
            targets[i, :y] = 1.0
    return F.binary_cross_entropy_with_logits(logits, targets)

def weighted_precision_recall(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.size == 0:
        return 0.0, 0.0
    n_classes = int(np.max(y_true)) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        precisions = np.diag(cm) / np.sum(cm, axis=0)
        recalls = np.diag(cm) / np.sum(cm, axis=1)
    precisions[np.isnan(precisions)] = 0
    recalls[np.isnan(recalls)] = 0
    supports = np.sum(cm, axis=1)
    total = np.sum(supports)
    if total == 0:
        return 0.0, 0.0
    w_precision = np.average(precisions, weights=supports)
    w_recall = np.average(recalls, weights=supports)
    return float(w_precision), float(w_recall)

def quadratic_weighted_kappa(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return 1.0
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
    if n_classes > 1:
        for i in range(n_classes):
            for j in range(n_classes):
                weight_matrix[i][j] = ((i - j) / (n_classes - 1.0)) ** 2
    num = np.sum(conf_matrix * weight_matrix)
    den = np.sum(exp_matrix * weight_matrix)
    if den == 0:
        return 1.0
    kappa = 1.0 - num / den
    return float(kappa)

def print_classification_report(y_true, y_pred, class_names):
    """
    Prints a sklearn-style classification report and confusion matrix.
    y_true, y_pred: 1D arrays / lists of integer class indices
    class_names: list of class names in index order
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n_classes = len(class_names)

    if y_true.size == 0:
        print("No samples to report.")
        return

    # Per-class metrics
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), zero_division=0
    )

    # Nicely formatted table
    print("\nClassification Report")
    print("=" * 80)
    print(f"{'Class':<30}{'Precision':>10}{'Recall':>10}{'F1-score':>10}{'Support':>10}")
    print("-" * 80)
    for i, cname in enumerate(class_names):
        print(f"{cname:<30}{p[i]:10.4f}{r[i]:10.4f}{f1[i]:10.4f}{support[i]:10d}")
    print("-" * 80)

    # Averages
    macro_p = np.mean(p) if p.size > 0 else 0.0
    macro_r = np.mean(r) if r.size > 0 else 0.0
    macro_f1 = np.mean(f1) if f1.size > 0 else 0.0
    weighted_p = np.average(p, weights=support) if support.sum() > 0 else 0.0
    weighted_r = np.average(r, weights=support) if support.sum() > 0 else 0.0
    weighted_f1 = np.average(f1, weights=support) if support.sum() > 0 else 0.0
    print(f"{'macro avg':<30}{macro_p:10.4f}{macro_r:10.4f}{macro_f1:10.4f}{support.sum():10d}")
    print(f"{'weighted avg':<30}{weighted_p:10.4f}{weighted_r:10.4f}{weighted_f1:10.4f}{support.sum():10d}")

    # sklearn's classification_report for convenience
    print("\nDetailed sklearn.report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("=" * 80 + "\n")

def evaluate(model, loader, class_names=None):
    model.eval()
    correct = total = 0
    total_loss = 0.0
    all_preds = []
    all_labels = []
    if class_names is None:
        class_names = [str(i) for i in range(NUM_CLASSES)]

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Testing")
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)  # (B, K-1)
            probs = torch.sigmoid(logits)  # probabilities for each threshold
            # For ordinal scheme, predicted class = number of thresholds whose prob > 0.5
            preds = (probs > 0.5).sum(dim=1).long()
            preds = torch.clamp(preds, 0, model.num_classes - 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loss = ordinal_bce_loss(logits, labels)
            total_loss += float(loss.item())

    acc = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    all_preds = np.array(all_preds, dtype=int)
    all_labels = np.array(all_labels, dtype=int)
    precision, recall = weighted_precision_recall(all_labels, all_preds)
    qwk_score = quadratic_weighted_kappa(all_labels, all_preds)

    # Print summary
    print(f"\nTest Loss: {avg_loss:.4f}, Acc: {acc:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, QWK: {qwk_score:.4f}")

    # Print classification report and confusion matrix
    print_classification_report(all_labels, all_preds, class_names)

    return {'acc': acc, 'qwk': qwk_score, 'precision': precision, 'recall': recall}

# Run evaluation
test_metrics = evaluate(model, test_loader, class_names=classes)

def predict_image(model, image_path, classes, transform, device):
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logit = model(image_tensor)
        probs = torch.sigmoid(logit)
        pred_idx = (probs > 0.5).sum().item()
        pred_idx = min(pred_idx, len(classes) - 1)
        pred_class = classes[pred_idx]
        # A crude confidence: mean of threshold probabilities for predicted class
        conf = float(probs.mean().item()) if probs.numel() > 0 else 0.0
    return pred_class, conf

# Example usage of predict_image (uncomment and set path to test a single image)
# img_path = "/path/to/some/image.jpg"
# cls, conf = predict_image(model, img_path, classes, val_test_transform, DEVICE)
# print(f"Predicted: {cls} (conf {conf:.3f})")
