## 📝 README: Mayo Clinic Severity Classification Model

### 🌟 Overview
This document summarizes the performance of a machine learning model trained to classify image severity based on the **Mayo Clinic Scoring System** (Mayo 0, 1, 2, 3). The model, which was loaded from **epoch 43**, was tested on a dedicated dataset to evaluate its ability to accurately classify the severity levels.

### 📊 Model Performance Summary

The model achieved a high level of agreement with the true labels, as measured by the **Quadratic Weighted Kappa (QWK)** score, which is particularly relevant for ordinal classification tasks.

| Metric | Value |
| :--- | :--- |
| **Test Dataset Size** | 1130 samples |
| **Loaded Epoch** | 43 |
| **Best QWK (Training)** | 0.8285 |
| **Test Loss** | 0.2172 |
| **Test Accuracy (Acc)** | **73.54%** |
| **Test QWK** | **0.8291** |
| **Weighted Average Precision** | 0.7425 |
| **Weighted Average Recall** | 0.7354 |

---
### 📈 Classification Report by Class

This table shows the model's performance for each individual severity class.

| Class | Support | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Mayo 0** | 611 | 0.8662 | 0.8265 | **0.8459** |
| **Mayo 1** | 306 | 0.5809 | 0.5752 | 0.5780 |
| **Mayo 2** | 126 | 0.5267 | 0.6270 | 0.5725 |
| **Mayo 3** | 87 | 0.7553 | 0.8161 | 0.7845 |

**Key Observations:**
* The model performs **strongest** on the most common class, **Mayo 0**, and the least common class, **Mayo 3**, as indicated by the high F1-scores.
* Performance is notably **lower** for the intermediate classes, **Mayo 1** and **Mayo 2**, particularly in terms of **Precision** for Mayo 2 (0.5267).

---
### 📉 Confusion Matrix

The confusion matrix shows the number of samples correctly and incorrectly classified for each class.

| True Class (Row) $\rightarrow$ Predicted Class (Column) | Mayo 0 | Mayo 1 | Mayo 2 | Mayo 3 |
| :--- | :--- | :--- | :--- | :--- |
| **Mayo 0** | **505** | 99 | 6 | 1 |
| **Mayo 1** | 75 | **176** | 52 | 3 |
| **Mayo 2** | 3 | 25 | **79** | 19 |
| **Mayo 3** | 0 | 3 | 13 | **71** |

