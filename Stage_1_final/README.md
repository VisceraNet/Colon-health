# 🧠 ResEffFusion – Masked Medical Image Classification

Efficient and explainable hybrid fusion model for **gastrointestinal disease classification** using mask-aware training.

---

# 🏗️ Model Architecture

## 🔷 ResEffFusion (EffResFusion)

ResEffFusion combines two strong CNN backbones:

- **EfficientNet-B4**
- **ResNet-50**

### 🔬 Fusion Strategy

1. Extract final feature maps from both backbones  
2. Project both to **1024 channels**
3. Weighted fusion  

   ```
   F = 0.75 × EfficientNet + 0.25 × ResNet
   ```

4. BatchNorm → ReLU (non-inplace, Grad-CAM++ safe)
5. Global Average Pooling
6. Fully connected classifier

### 🎯 Design Goals

- High classification accuracy  
- Mask-aware learning  
- Strong generalization  
- Grad-CAM++ interpretability  

---

# 📂 Dataset

Masked medical image classification.

Each image has an associated mask:

```
<image_name>_mask.png
```

### 🏷️ Classes

```
0_normal
1_ulcerative_colitis
2_polyps
3_esophagitis
```

### 🧪 Mask Handling Strategy

Masked-out pixels are replaced with **ImageNet mean values** before normalization.

This ensures:
- No artificial zero bias  
- Stable backbone feature extraction  
- Proper masked region suppression  

---

# 📊 Final Test Performance

| Model            | Accuracy | Macro F1 | Macro ROC-AUC | Macro PR-AUC |
|------------------|----------|----------|---------------|--------------|
| ResNet50         | 0.9802   | 0.9816   | 0.9994        | 0.9985       |
| EfficientNet-B4  | 0.9829   | 0.9848   | 0.9995        | 0.9988       |
| ViT              | 0.8855   | 0.8974   | 0.9798        | 0.9563       |
| **EffResFusion** | **0.9847** | **0.9854** | **0.9997** | **0.9993** |

🏆 **Fusion model achieves best overall performance across all macro metrics.**

---

# 📈 Stage 1 – Visual Evaluation

Plots directory:

```
Stage_1_final/plots_comparing_models/
```

---

## 🔷 ResNet50

| Confusion Matrix | Normalized CM | Precision-Recall |
|------------------|---------------|------------------|
| ![](https://raw.githubusercontent.com/VisceraNet/Colon-health/final/Stage_1_final/plots_comparing_models/ResNet50_cm.png) | ![](https://raw.githubusercontent.com/VisceraNet/Colon-health/final/Stage_1_final/plots_comparing_models/ResNet50_cm_norm.png) | ![](https://raw.githubusercontent.com/VisceraNet/Colon-health/final/Stage_1_final/plots_comparing_models/ResNet50_pr.png) |

| ROC Curve |
|-----------|
| ![](https://raw.githubusercontent.com/VisceraNet/Colon-health/final/Stage_1_final/plots_comparing_models/ResNet50_roc.png) |

---

## 🔷 EfficientNet-B4

| Confusion Matrix | Normalized CM | Precision-Recall |
|------------------|---------------|------------------|
| ![](https://raw.githubusercontent.com/VisceraNet/Colon-health/final/Stage_1_final/plots_comparing_models/EfficientNet-B4_cm.png) | ![](https://raw.githubusercontent.com/VisceraNet/Colon-health/final/Stage_1_final/plots_comparing_models/EfficientNet-B4_cm_norm.png) | ![](https://raw.githubusercontent.com/VisceraNet/Colon-health/final/Stage_1_final/plots_comparing_models/EfficientNet-B4_pr.png) |

| ROC Curve |
|-----------|
| ![](https://raw.githubusercontent.com/VisceraNet/Colon-health/final/Stage_1_final/plots_comparing_models/EfficientNet-B4_roc.png) |

---

## 🔷 Vision Transformer (ViT)

| Confusion Matrix | Normalized CM | Precision-Recall |
|------------------|---------------|------------------|
| ![](https://raw.githubusercontent.com/VisceraNet/Colon-health/final/Stage_1_final/plots_comparing_models/ViT_cm.png) | ![](https://raw.githubusercontent.com/VisceraNet/Colon-health/final/Stage_1_final/plots_comparing_models/ViT_cm_norm.png) | ![](https://raw.githubusercontent.com/VisceraNet/Colon-health/final/Stage_1_final/plots_comparing_models/ViT_pr.png) |

| ROC Curve |
|-----------|
| ![](https://raw.githubusercontent.com/VisceraNet/Colon-health/final/Stage_1_final/plots_comparing_models/ViT_roc.png) |

---

## 🔷 Hybrid Model – ResEffFusion

| Confusion Matrix | Normalized CM | Precision-Recall |
|------------------|---------------|------------------|
| ![](https://raw.githubusercontent.com/VisceraNet/Colon-health/final/Stage_1_final/training_plots/confusion_matrix.png) | ![](https://raw.githubusercontent.com/VisceraNet/Colon-health/final/Stage_1_final/training_plots/confusion_matrix_normalized.png) | ![](https://raw.githubusercontent.com/VisceraNet/Colon-health/final/Stage_1_final/training_plots/pr_curves.png) |

| ROC Curve |
|-----------|
| ![](https://raw.githubusercontent.com/VisceraNet/Colon-health/final/Stage_1_final/training_plots/roc_curves.png) |

---

# 🚀 Training Features

- Mixed Precision Training (AMP)
- Cosine Learning Rate Scheduler
- Early Stopping
- Best checkpoint saving
- Full train / validation / test evaluation
- Confusion Matrix, ROC & PR curve generation
- Grad-CAM++ visualizations
- CSV metric export for analysis

---

# 📝 Key Observations

- CNN-based models outperform ViT on masked GI dataset.
- EfficientNet-B4 slightly surpasses ResNet50.
- The weighted fusion strategy further improves macro metrics.
- Near-perfect ROC-AUC confirms strong separability.
- Grad-CAM++ highlights medically meaningful attention regions.

---

# 📌 Summary

**ResEffFusion** is a high-performance, mask-aware hybrid architecture combining EfficientNet-B4 and ResNet-50.

It achieves:

> 🎯 **98.47% Accuracy**  
> 🎯 **98.54% Macro F1**  
> 🎯 **0.9997 ROC-AUC**  

with strong interpretability via **Grad-CAM++**.

Designed for robust and explainable gastrointestinal disease classification.

---
