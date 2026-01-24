# test.py

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from timeit import default_timer as timer

from datasets import LIMUCDataset
from models import EFFResNetViT
from evaluate import evaluate
from config import *

def test():
    BASE_DIR = Path(__file__).resolve().parent.parent
    TEST_DIR = BASE_DIR / "data/test_set"

    print("Loading test dataset from:", TEST_DIR)

    test_ds = LIMUCDataset(str(TEST_DIR), train=False)

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = EFFResNetViT().to(DEVICE)

    ckpt_path = f"{CKPT_DIR}/best_model.pt"
    print("Loading checkpoint:", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    torch.cuda.synchronize()
    t0 = timer()

    metrics = evaluate(model, test_loader, DEVICE)

    torch.cuda.synchronize()
    t1 = timer()

    print(f"\nTest evaluation time: {t1 - t0:.2f}s")

    print("\n=== TEST SET SANITY CHECK ===")
    print(f"Recall (active): {metrics['recall_active']:.3f}")
    print(f"FNR:             {metrics['fnr']:.3f}")
    print(f"Precision:       {metrics['precision']:.3f}")
    print(f"Specificity:     {metrics['specificity']:.3f}")
    print(f"Accuracy:        {metrics['accuracy']:.3f}")
    print(f"QWK:             {metrics['qwk']:.3f}")
    print("Confusion Matrix:", metrics["confusion_matrix"])

    # ---- Optional numeric sanity peek
    with torch.no_grad():
        batch = next(iter(test_loader))
        imgs = batch["image"].to(DEVICE)

        out = model(imgs)

        print("\nSample binary probabilities:",
              torch.sigmoid(out["binary_logits"][:5]).cpu().numpy())
        print("Sample severity scores:",
              out["severity_score"][:5].cpu().numpy())


if __name__ == "__main__":
    test()
