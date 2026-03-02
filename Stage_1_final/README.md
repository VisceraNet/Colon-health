# ResEffFusion – Masked Medical Image Classification

Efficient and explainable fusion model for gastrointestinal disease classification using masked images.

---

## Model Architecture

**ResEffFusion** combines:

* **EfficientNet-B4**
* **ResNet-50**

Fusion strategy:

* Extract final feature maps from both backbones
* Project to 1024 channels
* Weighted fusion (`EFF_WEIGHT = 0.75` default)
* BatchNorm → ReLU (non-inplace for Grad-CAM++)
* Global Average Pooling → Linear classifier

Designed for:

* High accuracy
* Mask-aware training
* Grad-CAM++ interpretability

---

## Dataset

Masked image classification (mask file: `<image>_mask.png`)

Classes:

```
0_normal
1_ulcerative_colitis
2_polyps
3_esophagitis
```

Masked-out pixels are replaced with ImageNet mean before normalization.

---

## Final Test Performance

| Model            | Accuracy   | Macro F1   | ROC-AUC    | PR-AUC     |
| ---------------- | ---------- | ---------- | ---------- | ---------- |
| ResNet50         | 0.9802     | 0.9816     | 0.9994     | 0.9985     |
| EfficientNet-B4  | 0.9829     | 0.9848     | 0.9995     | 0.9988     |
| ViT              | 0.8855     | 0.8974     | 0.9798     | 0.9563     |
| **EffResFusion** | **0.9847** | **0.9854** | **0.9997** | **0.9993** |

Fusion model achieves the best overall performance.

---

## Training Features

* Mixed Precision (AMP)
* Cosine LR Scheduler
* Early Stopping
* Best checkpoint saving
* Full train/val/test evaluation
* Confusion matrix, ROC & PR curves
* Grad-CAM++ visualizations
* CSV metrics export

---
## Summary

ResEffFusion is a high-performance, mask-aware fusion network combining EfficientNet-B4 and ResNet-50, achieving **98.47% accuracy** with strong interpretability via Grad-CAM++.
