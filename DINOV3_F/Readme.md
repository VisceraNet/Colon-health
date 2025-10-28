## **I. Overview and Setup**

| Detail | Value |
| :--- | :--- |
| **Model Environment** | `dinov3_env` |
| **Evaluation Script** | `/home/phil/evaluation_archive.py` |
| **Device Used** | `cuda` |
| **Backbone State** | Frozen |
| **Loaded Checkpoint** | Epoch 47 |

---

## **II. Test Dataset Statistics**

The model was evaluated on a balanced test dataset of **800 samples**.

| Class ID | Class Name | Samples (Support) | Distribution |
| :--- | :--- | :--- | :--- |
| 0 | **0\_normal** | 200 | Balanced |
| 1 | **1\_ulcerative\_colitis** | 200 | Balanced |
| 2 | **2\_polyps** | 200 | Balanced |
| 3 | **3\_esophagitis** | 200 | Balanced |

---

## **III. Overall Model Performance**

The model demonstrates strong overall performance with an average accuracy exceeding 95%.

| Metric | Value |
| :--- | :--- |
| **Test Accuracy (Overall)** | **95.50%** |
| **Macro Average F1-score** | 0.9548 |
| **Weighted Average F1-score** | 0.9548 |

---

## **IV. Detailed Classification Report**

The table below breaks down performance metrics for each of the four classes.

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0\_normal** | 0.9217 | **1.0000** | 0.9592 | 200 |
| **1\_ulcerative\_colitis** | 0.9588 | 0.9300 | 0.9442 | 200 |
| **2\_polyps** | 0.9579 | 0.9100 | 0.9333 | 200 |
| **3\_esophagitis** | **0.9849** | 0.9800 | **0.9825** | 200 |



### **Per-Class Accuracy**

| Class | Accuracy |
| :--- | :--- |
| **0\_normal** | **100.00%** |
| **1\_ulcerative\_colitis** | 93.00% |
| **2\_polyps** | 91.00% |
| **3\_esophagitis** | 98.00% |



### **Confusion Matrix**

This matrix shows the true vs. predicted classifications, where the rows represent the true class and the columns represent the predicted class.

| True Class (Row) | Pred: 0\_normal | Pred: 1\_ulcerative\_colitis | Pred: 2\_polyps | Pred: 3\_esophagitis | **Total (Support)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0\_normal** | **200** | 0 | 0 | 0 | 200 |
| **1\_ulcerative\_colitis** | 4 | **186** | 8 | 2 | 200 |
| **2\_polyps** | 10 | 7 | **182** | 1 | 200 |
| **3\_esophagitis** | 3 | 1 | 0 | **196** | 200 |
