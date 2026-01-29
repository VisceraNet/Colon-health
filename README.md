# 🏥 AI-Based Gastrointestinal Disease Analysis System

## 📌 Overview
This project presents a **comprehensive AI framework for gastrointestinal (GI) endoscopy analysis**, combining:

- **Disease classification**
- **Ordinal severity prediction (Mayo score)**
- **Polyp segmentation using YOLOv11**

The system is designed for **research and academic evaluation**, emphasizing **clinical relevance, interpretability, and robust evaluation metrics**.

---

## 🧠 Model Components

### 1️⃣ Disease Classification
- Task: Multi-class classification  
- Classes: Normal, Ulcerative Colitis, Polyps, Esophagitis  
- Model: **EFFResNet-ViT**
  - EfficientNet-B4 → texture & mucosal details  
  - ResNet-50 → structural features  
  - Transformer → global context  
- Metrics: Accuracy, Precision, Recall, Specificity

---

### 2️⃣ Severity Prediction (Ordinal)
- Task: **Ordinal classification of Mayo scores (0–3)**
- Model: **EFFResNet-ViT with ordinal head**
- Evaluation:
  - **Quadratic Weighted Kappa (QWK)**  
  - Sensitivity & Specificity  
  - Confusion Matrix  
- Designed to reflect **clinical severity progression**, not simple class prediction.

---

### 3️⃣ Polyp Segmentation
- Task: Pixel-level polyp localization  
- Model: **YOLOv11 (Segmentation)**
- Purpose:
  - Accurate polyp boundary detection  
  - Supports clinical inspection and downstream analysis  
- Output: Segmentation masks over endoscopic images

---

## 🔍 Explainable AI (XAI)
- Method: **Score-CAM**
- Applied to both classification and severity models
- Provides **stable, clinically interpretable visual explanations**

---

## ⚠️ Disclaimer
This project is intended **for research and educational purposes only** and is **not approved for clinical diagnosis**.

---

## 🚀 Key Contribution
A unified AI pipeline that integrates **classification, ordinal severity assessment, and segmentation**, supported by **medical-grade evaluation and explainability**.
