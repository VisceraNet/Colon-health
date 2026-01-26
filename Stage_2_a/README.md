# 🏥 Ordinal Severity Classification (Mayo Score 0–3)
---
## 🧠 Model

* **Backbone:** EfficientNet-B4 + ResNet-50
* **Fusion:** Feature concatenation + lightweight Transformer encoder
* **Head:** Ordinal classification head (cumulative logits)
* **Loss:** Binary Cross-Entropy on ordinal boundaries
* **Primary metric:** **Quadratic Weighted Kappa (QWK)**

---

## 📂 Dataset

Endoscopic images organized into four ordinal classes:

```
Mayo_0, Mayo_1, Mayo_2, Mayo_3
```

Evaluation performed on an independent test set (n = 1686).

---

## 📊 Test Set Performance

### 🔹 Classification Report

| Class                | Precision | Recall (Sensitivity) | F1-score | Support |
| -------------------- | --------- | -------------------- | -------- | ------- |
| Mayo_0               | 0.86      | 0.87                 | 0.87     | 925     |
| Mayo_1               | 0.66      | 0.64                 | 0.65     | 464     |
| Mayo_2               | 0.53      | 0.64                 | 0.58     | 177     |
| Mayo_3               | 0.67      | 0.47                 | 0.55     | 120     |
| **Overall Accuracy** |           |                      | **0.75** | 1686    |

---

### 🔹 Sensitivity & Specificity (Medical Metrics)

| Class  | Sensitivity | Specificity |
| ------ | ----------- | ----------- |
| Mayo_0 | 0.8735      | 0.8305      |
| Mayo_1 | 0.6358      | 0.8732      |
| Mayo_2 | 0.6384      | 0.9324      |
| Mayo_3 | 0.4667      | 0.9821      |



---

