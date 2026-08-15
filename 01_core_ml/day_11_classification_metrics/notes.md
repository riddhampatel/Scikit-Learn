# 📊 Day 11: Classification Metrics

## 1. Overview & Key Concepts

When evaluating classification models, **Accuracy alone is rarely sufficient**—especially when dealing with imbalanced datasets (e.g., fraud detection where 99% of transactions are legitimate). Understanding complementary metrics like **Precision**, **Recall**, **F1-Score**, and **ROC-AUC** is essential for selecting and tuning models based on specific real-world consequences of prediction errors.

---

## 2. The Confusion Matrix

The **Confusion Matrix** forms the foundation of all classification metrics. For binary classification:

| | Predicted Negative (0) | Predicted Positive (1) |
|---|---|---|
| **Actual Negative (0)** | **TN** (True Negative) | **FP** (False Positive / Type I Error) |
| **Actual Positive (1)** | **FN** (False Negative / Type II Error) | **TP** (True Positive) |

### Key Definitions:
* **True Positive (TP):** Model correctly predicted Positive.
* **True Negative (TN):** Model correctly predicted Negative.
* **False Positive (FP):** Model predicted Positive, but actual was Negative (*False Alarm / Type I Error*).
* **False Negative (FN):** Model predicted Negative, but actual was Positive (*Missed Detection / Type II Error*).

---

## 3. Core Metrics & Formulas

### 3.1 Accuracy
The proportion of total correct predictions.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

> ⚠️ **Warning (Accuracy Paradox):** On an imbalanced dataset where 95% of samples are Class 0, a dummy model that always predicts Class 0 gets 95% accuracy while completely failing to detect Class 1!

---

### 3.2 Precision (Exactness)
Out of all instances the model predicted as **Positive**, how many were **actually Positive**?

$$\text{Precision} = \frac{TP}{TP + FP}$$

* **Goal:** Minimize False Positives ($FP$).
* **When to prioritize Precision:**
  * **Spam Detection:** You don't want a non-spam email (important work email) sent to the Spam folder.
  * **Search Engine / Recommender Systems:** Users prefer relevant results over noisy/irrelevant suggestions.
  * **Legal / Drug Testing:** False positives have severe penalties or reputational damage.

---

### 3.3 Recall / Sensitivity / True Positive Rate (Completeness)
Out of all **actual Positive** instances in the dataset, how many did the model **successfully catch**?

$$\text{Recall} = \frac{TP}{TP + FN}$$

* **Goal:** Minimize False Negatives ($FN$).
* **When to prioritize Recall:**
  * **Medical Diagnosis (e.g., Cancer Screening):** Missing a malignant tumor ($FN$) can be fatal; a false alarm ($FP$) just leads to follow-up testing.
  * **Fraud Detection:** Missing a fraudulent transaction ($FN$) causes direct financial loss.
  * **Security & Threat Detection:** Missing an intruder or cyber threat ($FN$) compromises systems.

---

### 3.4 F1-Score
The **harmonic mean** of Precision and Recall. It provides a single balanced metric when you need a balance between Precision and Recall.

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

> Note: Arithmetic mean can be misleading (e.g., Precision=1.0, Recall=0.0 gives arithmetic mean of 0.5, but F1=0.0).

---

### 3.5 Multi-Class Averaging Strategies
For datasets with $>2$ classes, Scikit-Learn provides averaging options:
* **`macro`:** Unweighted mean of metrics across all classes (treats all classes equally, great for highlighting performance on minority classes).
* **`weighted`:** Mean weighted by support (number of true instances for each class).
* **`micro`:** Total $TP, FP, FN$ pooled globally (equals accuracy in single-label multi-class classification).

---

## 4. Threshold Tuning & Precision-Recall Tradeoff

Logistic Regression outputs probabilities between 0 and 1. By default, Scikit-Learn uses a decision threshold of **0.5**:

$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|X) \ge 0.5 \\ 0 & \text{if } P(y=1|X) < 0.5 \end{cases}$$

* **Lowering Threshold (e.g., to 0.2):**
  * Model predicts Positive more aggressively $\rightarrow$ **Increases Recall**, **Decreases Precision**.
* **Raising Threshold (e.g., to 0.8):**
  * Model predicts Positive only when highly confident $\rightarrow$ **Increases Precision**, **Decreases Recall**.

---

## 5. ROC Curve & ROC-AUC Score

### 5.1 ROC Curve (Receiver Operating Characteristic)
Plots **True Positive Rate (Recall)** against **False Positive Rate (FPR)** across all possible probability thresholds ($0.0$ to $1.0$).

$$\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$

### 5.2 ROC-AUC Score (Area Under Curve)
* **AUC = 1.0:** Perfect classifier.
* **AUC = 0.5:** Random guessing (diagonal line).
* **Interpretation:** Probability that the model ranks a randomly chosen positive sample higher than a randomly chosen negative sample.

---

## 6. Scikit-Learn API Cheat Sheet

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
)

# Core Metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

# Comprehensive Report
print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))

# ROC-AUC (requires probabilities, not class labels!)
y_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_true, y_proba)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_proba)
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score).plot()
```

---

## 7. Summary & Decision Matrix

| Problem Scenario | Primary Error to Avoid | Key Metric |
|---|---|---|
| Medical Disease Detection | False Negative ($FN$) | **Recall** |
| Email Spam Filter | False Positive ($FP$) | **Precision** |
| Balanced Dataset Evaluation | General errors | **Accuracy / F1-Score** |
| Imbalanced Class Ranking | Threshold-agnostic ranking | **ROC-AUC / PR-AUC** |
