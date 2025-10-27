# Classification Model Evaluation Report: Gastrointestinal Endoscopy Images

This report summarizes the performance of the classification model trained on gastrointestinal endoscopy images, evaluated on a held-out test set.

## 🚀 Overview and Key Results

The model achieved an overall **Test Accuracy of $93.44\%$** across 14 distinct classes of gastrointestinal findings. The evaluation was conducted on a balanced test set, with **3,435 samples per class**, totaling **48,090 samples**. The model's backbone was frozen during the evaluation, and the checkpoint loaded was from **epoch 45**.

---

## 📊 Overall Performance Metrics

The macro and weighted averages indicate a high and consistent level of performance across the board.

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | $0.9344$ ($93.44\%$) |
| **Macro Average Precision** | $0.9343$ |
| **Macro Average Recall** | $0.9344$ |
| **Macro Average F1-score** | $0.9342$ |
| **Weighted Average Precision** | $0.9343$ |
| **Weighted Average Recall** | $0.9344$ |
| **Weighted Average F1-score** | $0.9342$ |
| **Total Test Samples** | $48,090$ |

---

## 🎯 Per-Class Performance Breakdown

Performance varied across classes, with some showing near-perfect classification and others presenting greater challenges.

### High-Performing Classes (F1-score $\ge 0.99$)

These classes were classified with extremely high accuracy and confidence:

| Class | Precision | Recall | F1-score | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **blood\_hematin.tar** | $1.0000$ | $1.0000$ | $1.0000$ | $100.00\%$ |
| **polyp.tar** | $0.9977$ | $1.0000$ | $0.9988$ | $100.00\%$ |
| **ampulla\_of\_vater.tar** | $0.9997$ | $1.0000$ | $0.9999$ | $100.00\%$ |

### Challenging Classes (F1-score $\le 0.88$)

These classes represent areas where the model could potentially be improved, particularly **normal\_clean\_mucosa.tar**, which had the lowest recall.

| Class | Precision | Recall | F1-score | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **normal\_clean\_mucosa.tar** | $0.8367$ | $0.7828$ | $0.8088$ | $78.28\%$ |
| **erosion.tar** | $0.8550$ | $0.8512$ | $0.8531$ | $85.12\%$ |
| **pylorus.tar** | $0.8537$ | $0.8934$ | $0.8731$ | $89.34\%$ |
| **ileocecal\_valve.tar** | $0.8751$ | $0.8897$ | $0.8823$ | $88.97\%$ |

### Full Classification Report

| Class | Precision | Recall | F1-score | Support | Per-Class Accuracy (Recall) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ampulla\_of\_vater.tar | $0.9997$ | $1.0000$ | $0.9999$ | $3435$ | $100.00\%$ |
| angiectasia.tar | $0.9416$ | $0.9197$ | $0.9305$ | $3435$ | $91.97\%$ |
| blood\_fresh.tar | $0.9810$ | $0.9918$ | $0.9864$ | $3435$ | $99.18\%$ |
| blood\_hematin.tar | $1.0000$ | $1.0000$ | $1.0000$ | $3435$ | $100.00\%$ |
| erosion.tar | $0.8550$ | $0.8512$ | $0.8531$ | $3435$ | $85.12\%$ |
| erythema.tar | $0.9697$ | $0.9767$ | $0.9732$ | $3435$ | $97.67\%$ |
| foreign\_body.tar | $0.9716$ | $0.9377$ | $0.9544$ | $3435$ | $93.77\%$ |
| ileocecal\_valve.tar | $0.8751$ | $0.8897$ | $0.8823$ | $3435$ | $88.97\%$ |
| lymphangiectasia.tar | $0.9471$ | $0.9697$ | $0.9583$ | $3435$ | $96.97\%$ |
| **normal\_clean\_mucosa.tar** | **0.8367** | **0.7828** | **0.8088** | $3435$ | **78.28%** |
| polyp.tar | $0.9977$ | $1.0000$ | $0.9988$ | $3435$ | $100.00\%$ |
| pylorus.tar | $0.8537$ | $0.8934$ | $0.8731$ | $3435$ | $89.34\%$ |
| reduced\_mucosal\_view.tar | $0.9265$ | $0.9243$ | $0.9254$ | $3435$ | $92.43\%$ |
| ulcer.tar | $0.9245$ | $0.9441$ | $0.9342$ | $3435$ | $94.41\%$ |

---

## 🧐 Analysis of Misclassifications (Confusion Matrix)

The confusion matrix provides insight into which classes are being confused with one another. Each row represents the true class, and each column represents the predicted class. The indices correspond to the class list:

**Classes (Indices 0-13):**
0. ampulla\_of\_vater.tar
1. angiectasia.tar
2. blood\_fresh.tar
3. blood\_hematin.tar
4. erosion.tar
5. erythema.tar
6. foreign\_body.tar
7. ileocecal\_valve.tar
8. lymphangiectasia.tar
9. **normal\_clean\_mucosa.tar**
10. polyp.tar
11. **pylorus.tar**
12. **reduced\_mucosal\_view.tar**
13. ulcer.tar
