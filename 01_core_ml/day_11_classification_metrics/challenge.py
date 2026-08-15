"""
DAY 11 ACTIVE RECALL CHALLENGE: Classification Metrics
"""

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ===================================================================
# TASK 1: Load Dataset & Stratified Train/Test Split
# ===================================================================

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ===================================================================
# TASK 2: Feature Scaling
# ===================================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ===================================================================
# TASK 3: Model Training & Predictions
# ===================================================================

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

y_proba_class_1 = model.predict_proba(X_test_scaled)[:, 1]


# ===================================================================
# TASK 4: Calculate Individual Classification Metrics
# ===================================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_proba_class_1)


print("=" * 60)
print("EVALUATION METRICS RESULTS")
print("=" * 60)
print(f"Accuracy:  {accuracy}")
print(f"Precision: {precision}")
print(f"Recall:    {recall}")
print(f"F1-Score:  {f1}")
print(f"ROC-AUC:   {roc_auc}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ===================================================================
# TASK 5: Custom Probability Thresholding (Recall vs Precision Tuning)
# ===================================================================

# Goal: Lower the decision threshold to 0.30 to prioritize catching malignant cases

y_pred_custom_thresh = (y_proba_class_1 >= 0.30).astype(int)

precision_30 = precision_score(y_test, y_pred_custom_thresh)

recall_30 = recall_score(y_test, y_pred_custom_thresh)


print("\n" + "=" * 60)
print("CUSTOM THRESHOLD (0.30) EVALUATION")
print("=" * 60)
print(f"Precision at 0.30 threshold: {precision_30}")
print(f"Recall at 0.30 threshold:    {recall_30}")