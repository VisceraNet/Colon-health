# Kvasir Classification — README

**Brief:**
This repository contains a fast supervised classifier trained on your prepared Kvasir dataset (split into `train/ val/ test/`). The best checkpoint `best.pth` achieves **Overall accuracy: 0.864338** on the test split. This README summarizes results, how to reproduce evaluation, how to run inference on new images, and quick tips to improve performance.

---

# Results (test set)

**Overall accuracy:** `0.864338`
**Total test samples:** `48,090` (14 classes × 3,435 each)

Per-class summary (precision / recall / f1 / support):

```
ampulla_of_vater.tar        0.99  1.00  1.00  3435
angiectasia.tar            0.92  0.83  0.87  3435
blood_fresh.tar            0.96  0.96  0.96  3435
blood_hematin.tar          0.99  1.00  0.99  3435
erosion.tar                0.74  0.66  0.70  3435
erythema.tar               0.83  0.93  0.88  3435
foreign_body.tar           0.86  0.85  0.86  3435
ileo-cecal_valve.tar       0.84  0.74  0.79  3435
lymphangiectasia.tar       0.84  0.90  0.87  3435
normal_clean_mucosa.tar    0.72  0.66  0.69  3435
polyp.tar                  0.95  1.00  0.97  3435
pylorus.tar                0.74  0.83  0.78  3435
reduced_mucosal_view.tar   0.81  0.89  0.85  3435
ulcer.tar                  0.90  0.83  0.87  3435
```

**Macro / weighted avg:** ~0.86 precision / recall / f1.

**Observations**

* Strong performance on `ampulla_of_vater`, `blood_hematin`, `polyp`.
* Lower recall on `erosion` and `normal_clean_mucosa` (possible class confusion / visual similarity).
* Dataset is perfectly balanced (each class has same support) — metrics are not biased by class imbalance.

---

# Files & scripts

* `train_kvasir_classifier.py` — training script used (backbone freeze by default; optional top-k unfreeze).
* `train_dinov3_final.py` — earlier unsupervised DINOv3 training scripts (kept for reference).
* `eval_kvasir_best.py` — inference script that loads `best.pth` and writes per-image predictions to CSV.
* `make_classification_report.py` — generates classification report, per-class CSV, confusion matrix PNG/CSV from predictions CSV.
* `prepare_kvasir_classes.py` — dataset extraction / augmentation / split script you ran earlier.
* `best.pth` — checkpoint produced during training (placed under `out_dir` used for training).

---

# Quick environment (recommended)

Create the Python environment similar to what was used:

```bash
python -m venv dinov3_env
source dinov3_env/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or appropriate CUDA wheel
pip install timm pillow tqdm matplotlib seaborn scikit-learn pandas
```

(Adjust the `torch` wheel for your CUDA version. You used CUDA 13.0 on RTX 4080 SUPER.)

---

# How to evaluate `best.pth` on the test split

1. Run inference producing `predictions.csv`:

```bash
python3 eval_kvasir_best.py \
  --test_root /home/phil/kvasir_prepared/split/test \
  --ckpt /mnt/d/kvasir_cls_out/best.pth \
  --out_csv /mnt/d/kvasir_preds.csv \
  --batch_size 128 \
  --workers 8 \
  --amp
```

2. Produce a classification report + confusion matrix:

```bash
python3 make_classification_report.py \
  --csv /mnt/d/kvasir_preds.csv \
  --out_dir /mnt/d/kvasir_report \
  --plot
```

Outputs: `/mnt/d/kvasir_report/per_class_metrics.csv`, `confusion_matrix.csv`, `confusion_matrix.png`, `summary.txt`.

---

# How to run on a folder of unseen images (no labels)

```bash
python3 eval_kvasir_best.py \
  --test_root /home/phil/unseen_images \
  --ckpt /mnt/d/kvasir_cls_out/best.pth \
  --out_csv /mnt/d/kvasir_unseen_preds.csv \
  --batch_size 128 \
  --workers 8 \
  --amp
```

`kvasir_unseen_preds.csv` will contain rows: `path,pred_idx,pred_label,prob`.

---

# How to fine-tune further (if you want more accuracy)

Options (do one at a time and monitor val accuracy):

1. **Unfreeze top blocks** (fine-tune last transformer/conv blocks):
   Example — unfreeze top 2 blocks:

   ```bash
   python3 train_kvasir_classifier.py \
     --data_root /home/phil/kvasir_prepared/split \
     --out_dir /mnt/d/kvasir_cls_ft \
     --backbone convnext_tiny \
     --image_size 224 \
     --batch_size 48 \
     --epochs 10 \
     --workers 10 \
     --amp \
     --unfreeze_top_k 2 \
     --resume /mnt/d/kvasir_cls_out/best.pth
   ```

   Use smaller batch (e.g., 48) when training more params.

2. **Train longer / use cosine annealing** — increase epochs and use a learning-rate schedule.

3. **Stronger augmentation** — rotate, random crop scales, color jitter ranges specific to capsule imagery.

4. **Ensemble** — train few classifiers with different seeds/backbones and average predictions.

5. **Class-specific reweighting / calibration** — for classes with low recall (e.g., `erosion`), consider sampling augmentation or small per-class oversampling.

---

# Notes about checkpoint formats & loading

* `eval_kvasir_best.py` is robust and will attempt to read common checkpoint formats. If you used a different naming scheme for the saved head/backbone keys, give the script the exact path to `best.pth`.
* If you need me to convert or inspect `best.pth`, paste the output of:

```python
python - <<'PY'
import torch, sys
ck = torch.load("/path/to/best.pth", map_location="cpu")
print(type(ck))
print(list(ck.keys())[:50])
PY
```

and I’ll tell you how to map keys.

---

# Quick interpretation & next steps (short)

* **Good baseline**: 86.4% on a balanced, fairly large test set — solid for a linear probe on a frozen backbone.
* **Highest payoff**: fine-tune top blocks or focus on improving recall for `erosion` and `normal_clean_mucosa` via targeted augmentation or more curated examples.
* **If inference speed or memory is a concern**: convert model to `torch.jit.trace` or use `onnx`/`triton` after validation.

---

# Contact / reproducibility notes

* Commands in this README assume paths used on your machine (adjust if different).
* If you want, I can:

  * generate a small HTML report that includes `confusion_matrix.png` and metrics,
  * run class-wise error analysis (example images for confusions),
  * or create a short script to infer on a directory and copy misclassified images to a folder for review.

Which of those would you like next?
