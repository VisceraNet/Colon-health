# 🏥 EFFResNet-ViT for Gastrointestinal Disease Classification

---

## 🧠 Model Architecture

* **EfficientNet-B4** → fine mucosal texture & lesion details
* **ResNet-50** → robust structural and shape features
* **Vision Transformer** → global contextual reasoning
* **Fusion + Transformer Encoder** → final decision making


## 📊 Evaluation Metrics (Test Set)

### 🔹 Classification Report

| Class                | Precision | Recall (Sensitivity) | F1-Score |
| -------------------- | --------- | -------------------- | -------- |
| Normal               | **0.99**  | **1.00**             | **1.00** |
| Ulcerative Colitis   | 0.95      | 0.89                 | 0.92     |
| Polyps               | 0.90      | 0.94                 | 0.92     |
| Esophagitis          | **0.98**  | **1.00**             | **0.99** |
| **Overall Accuracy** |           |                      | **95%**  |

---

### 🔹 Sensitivity & Specificity (Medical Metrics)

| Class              | Sensitivity | Specificity |
| ------------------ | ----------- | ----------- |
| Normal             | **1.0000**  | 0.9967      |
| Ulcerative Colitis | 0.8850      | 0.9850      |
| Polyps             | ~0.94       | ~0.97       |
| Esophagitis        | **1.0000**  | 0.9933      |

✅ High **sensitivity** ensures diseases are not missed
✅ High **specificity** minimizes false positives
✅ Balanced performance across all GI conditions

