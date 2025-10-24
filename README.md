# Kvasir Classification


# Test set results

**Overall accuracy:** **`86.4338%`**
**Total test samples:** **48,090** (14 classes × 3,435 each)

|                    Class | Precision | Recall | F1-score | Support |
| -----------------------: | :-------: | :----: | :------: | ------: |
|     ampulla_of_vater.tar |    0.99   |  1.00  |   1.00   |   3,435 |
|          angiectasia.tar |    0.92   |  0.83  |   0.87   |   3,435 |
|          blood_fresh.tar |    0.96   |  0.96  |   0.96   |   3,435 |
|        blood_hematin.tar |    0.99   |  1.00  |   0.99   |   3,435 |
|              erosion.tar |    0.74   |  0.66  |   0.70   |   3,435 |
|             erythema.tar |    0.83   |  0.93  |   0.88   |   3,435 |
|         foreign_body.tar |    0.86   |  0.85  |   0.86   |   3,435 |
|      ileocecal_valve.tar |    0.84   |  0.74  |   0.79   |   3,435 |
|     lymphangiectasia.tar |    0.84   |  0.90  |   0.87   |   3,435 |
|  normal_clean_mucosa.tar |    0.72   |  0.66  |   0.69   |   3,435 |
|                polyp.tar |    0.95   |  1.00  |   0.97   |   3,435 |
|              pylorus.tar |    0.74   |  0.83  |   0.78   |   3,435 |
| reduced_mucosal_view.tar |    0.81   |  0.89  |   0.85   |   3,435 |
|                ulcer.tar |    0.90   |  0.83  |   0.87   |   3,435 |

**Summary (macro / weighted averages):** ~**0.86** precision / recall / f1.



# Important commands

> Replace paths with your actual locations (example paths below match earlier conversation).

## 1. Environment & dependencies

Create a Python virtualenv and install required packages:

```bash
python -m venv dinov3_env
source dinov3_env/bin/activate
pip install --upgrade pip
# Choose the appropriate torch wheel for your CUDA version; example for CUDA 13 (adjust if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm pillow tqdm matplotlib seaborn scikit-learn pandas
```

## 2. Run evaluation on labeled test set (produce predictions CSV)

```bash
python3 eval_kvasir_best.py \
  --test_root /home/phil/kvasir_prepared/split/test \
  --ckpt /mnt/d/kvasir_cls_out/best.pth \
  --out_csv /mnt/d/kvasir_preds.csv \
  --batch_size 128 \
  --workers 8 \
  --amp
```

## 3. Evaluate unlabeled folder (get predictions)

```bash
python3 eval_kvasir_best.py \
  --test_root /home/phil/unseen_images \
  --ckpt /mnt/d/kvasir_cls_out/best.pth \
  --out_csv /mnt/d/kvasir_unseen_preds.csv \
  --batch_size 128 \
  --workers 8 \
  --amp
```

## 4. Produce classification report & confusion matrix from CSV

```bash
python3 make_classification_report.py \
  --csv /mnt/d/kvasir_preds.csv \
  --out_dir /mnt/d/kvasir_report \
  --plot
# outputs: per_class_metrics.csv, confusion_matrix.csv, confusion_matrix.png, summary.txt
```

## 5. Quick inspect checkpoint contents (helpful if keys are unexpected)

```bash
python - <<'PY'
import torch, sys
ck = torch.load("/mnt/d/kvasir_cls_out/best.pth", map_location="cpu")
print(type(ck))
print(list(ck.keys())[:50])
PY
```

## 6. Resume training (if you later want to fine-tune)

Example: unfreeze last 2 blocks, smaller batch for fine-tuning

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

## 7. Run training (full run) — safe command (example)

```bash
tmux new -s kvasir_train -d "python3 train_kvasir_classifier.py \
  --data_root /home/phil/kvasir_prepared/split \
  --out_dir /mnt/d/kvasir_cls_out \
  --backbone convnext_tiny \
  --image_size 224 \
  --batch_size 96 \
  --epochs 20 \
  --workers 12 \
  --prefetch_factor 8 \
  --amp \
  --no_torch_compile"
```

## 8. Manage tmux session

Attach to see live output:

```bash
tmux attach -t kvasir_train
# detach: Ctrl-B then D
```

List sessions:

```bash
tmux ls
```

Kill session:

```bash
tmux kill-session -t kvasir_train
```

## 9. Start logging tmux output to file (no restart)

```bash
tmux pipe-pane -t kvasir_train 'cat >> /home/phil/kvasir_train.log'
# stop logging:
tmux pipe-pane -t kvasir_train
# view logs:
tail -f /home/phil/kvasir_train.log
```

## 10. Monitor GPU / processes

```bash
nvidia-smi         # GPU usage & processes
ps aux | grep python
htop               # interactive CPU/memory monitor
free -h            # RAM check
```

## 11. Force-recover WSL (if it becomes unresponsive)

> Use only if session hangs or system becomes unresponsive:

```powershell
# run from Windows PowerShell (not WSL)
wsl --shutdown
# reopen Ubuntu afterwards
```

## 12. Convert trained head to ONNX (optional - for deployment)

```bash
python - <<'PY'
import torch
ck = torch.load("/mnt/d/kvasir_cls_out/best.pth", map_location="cpu")
# load model forward function & head as in eval script, then:
# dummy = torch.randn(1,3,224,224).to("cpu")
# torch.onnx.export(model_fn, dummy, "model.onnx", opset_version=13)
PY
```

(See `eval_kvasir_best.py` for how to construct the backbone forward function; convert only after validating predictions.)

---

## Quick troubleshooting tips

* If you hit OOM, reduce `--batch_size` (e.g., 96 → 64 → 48) or reduce `--workers`.
* If WSL freezes: use tmux, avoid `torch.compile` on WSL initially, and use `wsl --shutdown` if you must kill the VM.


---



