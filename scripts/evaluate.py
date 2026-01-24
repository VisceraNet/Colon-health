# evaluate.py

import torch
from sklearn.metrics import cohen_kappa_score
from torch.amp import autocast



def evaluate(model, dataloader, device, threshold=0.4):
    """
    Evaluation focused on Active disease detection.
    
    Positive class = Active (Mayo 2–3)
    Negative class = Remission (Mayo 0–1)
    """
    # MAX_EVAL_BATCHES = 100
    model.eval()


    TP = FP = TN = FN = 0

    # For QWK (ordinal agreement)
    all_true_ord = []
    all_pred_ord = []

    with torch.no_grad():
    
        # for i, batch in enumerate(dataloader):
        #     if i >= MAX_EVAL_BATCHES:
        #         break

        for batch in dataloader:
            imgs = batch["image"].to(device)
            gt = batch["binary"].to(device)   # 0 = remission, 1 = active

            # with autocast(device_type="cuda"):
            out = model(imgs)


            # ---------- QWK (ordinal severity) ----------
            severity = out["severity_score"]

            pred_mayo = torch.clamp(
                torch.round(severity),
                min=0,
                max=3
            )

            all_true_ord.extend(batch["ordinal"].cpu().tolist())
            all_pred_ord.extend(pred_mayo.cpu().tolist())


            probs = torch.sigmoid(
                out["binary_logits"].view(-1)
            )

            preds = (probs >= threshold).long()

            TP += ((preds == 1) & (gt == 1)).sum().item()
            TN += ((preds == 0) & (gt == 0)).sum().item()
            FP += ((preds == 1) & (gt == 0)).sum().item()
            FN += ((preds == 0) & (gt == 1)).sum().item()

    # Metrics (safe against division by zero)
    recall = TP / (TP + FN + 1e-8)        # sensitivity (MOST IMPORTANT)
    fnr = FN / (TP + FN + 1e-8)            # false negative rate
    precision = TP / (TP + FP + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    # Quadratic Weighted Kappa (ordinal agreement)
    if len(all_true_ord) > 0:
        qwk = cohen_kappa_score(
            all_true_ord,
            all_pred_ord,
            weights="quadratic"
        )
    else:
        qwk = 0.0

    return {
        "recall_active": recall,
        "fnr": fnr,
        "precision": precision,
        "specificity": specificity,
        "accuracy": accuracy,
        "qwk": qwk,
        "confusion_matrix": {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN
        }
    }
