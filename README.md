# GI Tract Disease Severity Classification (LIMUC)

This repository contains a CNN–Transformer hybrid pipeline for classifying ulcerative colitis severity from endoscopic images using the LIMUC dataset.
The task is framed as a recall-prioritized binary classification (remission vs active) with an auxiliary ordinal ranking objective (Mayo 0–3).

The system is designed to be:

* Clinically safety-oriented (minimize false negatives)

* Ordinal-aware (severity is not categorical noise)

* Explainability-ready (post-hoc analysis on intermediate features)

---


## 1. Problem Definition
### Binary Task (Primary)

* Remission: Mayo 0–1

* Active disease: Mayo 2–3

#### Primary objective:

```Maximize recall for active disease (minimize false negatives)```

---


### Ordinal Task (Secondary)

* Predict relative severity ordering across Mayo scores (0–3)

* Used as a regularizing signal, not the primary decision

#### Evaluation metric:

* Quadratic Weighted Kappa (QWK)

---

## 2. Dataset

### LIMUC (Labeled Images for Ulcerative Colitis)

* Endoscopic images grouped into folders:

```
Mayo 0/
Mayo 1/
Mayo 2/
Mayo 3/
```


* Images resized to 224×224, RGB

* Binary labels derived as:

    * ```0``` → Mayo ≤ 1 (remission)

    * ```1``` → Mayo ≥ 2 (active)

No CSV annotations are used; labels are inferred from folder names.

---

## 3. Model Architecture
### Overview

#### EFFResNet-ViT: a CNN–Transformer hybrid with dual heads.

```
Input Image
   │
   ├── EfficientNet-B4 (CNN backbone)
   ├── ResNet-50 (CNN backbone)
   │
   └── Feature Fusion (channel-wise concat)
           │
           └── 1×1 Projection → Tokens
                   │
                   └── Transformer Encoder
                           │
                           ├── Binary Head (Active vs Remission)
                           └── Ordinal Head (Severity Score)

```

### Key Design Choices

* Dual CNN backbones → capture complementary texture + structure

* Transformer encoder → global context aggregation

* Two-task heads → safety-focused decision + ordinal regularization

* No explainability constraints during training

---

## 4. Training Setup
### Core Parameters

| Parameter |	Value  |
|-----------|----------|
|Image size|	224 × 224|
|Batch size	|8|
|Optimizer	|AdamW|
|Initial LR	| 1e-4 |
|Fine-tune LR|	1e-5|
|Weight decay |	1e-4|
|Epochs|	10|
|AMP	Enabled| (training only)|


### Freezing Strategy

* CNN backbones + transformer frozen for warm-up

* Fully unfrozen after initial epochs

---


### Loss Functions

#### Total Loss:

```L = λ_binary · BCE + λ_rank · ListNet```


* Binary loss:
  Weighted Binary Cross-Entropy (recall-biased)

* Ordinal loss:
  ListNet loss computed over batch rankings

#### Loss weights:

* ```λ_binary = 1.0```

* ```λ_rank = 0.5```

---

## 5. Evaluation Protocol

Evaluation is post-hoc, deterministic, and run on the full validation/test set.

### Binary Metrics

* Recall (Active) ← primary

* False Negative Rate (FNR)

* Precision

* Specificity

* Accuracy

* Confusion matrix

### Ordinal Metric

* Quadratic Weighted Kappa (QWK)
  Computed from predicted severity scores (rounded & clamped to 0–3)

---

## 6. Training Results (Validation)

|Epoch|	Recall|	FNR|	Precision|	QWK|	Accuracy|
|-----|-------|----|-------------|-----|------------|
|1	|0.958	|0.042	|0.498	|0.439	|0.808|
|3	|0.964	|0.036	|0.565	|0.727	|0.852|
|10	|0.960	|0.040	|0.675	|0.545	|0.905|

#### Observations:

* Recall remains consistently high

* Ordinal signal peaks early, then stabilizes

* Precision improves with training

---

## 7. Test Set Results (Sanity Check)

### Unseen test set, same Mayo folder structure

|Metric	|Value|
|-------|-----|
|Recall (Active)|	0.936|
|FNR|	0.064|
|Precision|	0.552|
|Specificity|	0.837|
|Accuracy|	0.855|
|QWK|	0.440|

### Confusion Matrix:
```
TP: 278
FN: 19
FP: 226
TN: 1163
```


#### This confirms:

* Strong generalization for active disease detection

* Controlled false negatives

* Non-collapsed ordinal severity signal

---

## 8. Explainability (Planned)

Explainability is treated as a post-hoc analysis step, not a training constraint.

### Planned approach:

* Apply Layer-CAM on intermediate CNN feature maps

* Generate separate maps for:

    * Binary decision (active vs remission)

    * Ordinal severity prediction

* Use Score-CAM selectively as a validation tool

Explainability results are intended for qualitative inspection, not optimization.

---


## 9. Project Status

* ✅ Training pipeline stable

* ✅ Validation & test sanity checks passed

* ✅ Model checkpoint ready

* ⏳ Explainability analysis (next phase)

---

## 10. Notes for Contributors

* Do not tune thresholds before analysis

* Do not modify losses without re-running sanity checks

* Treat QWK as supporting evidence, not a primary objective

* Explainability should always be run on frozen checkpoints