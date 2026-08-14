"""
DAY 10 ACTIVE RECALL CHALLENGE: Logistic Regression (Binary Classification)
"""

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
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
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ===================================================================
# TASK 2: Feature Scaling
# ===================================================================

scaler = StandardScaler()

# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)


# ===================================================================
# TASK 3: Fit Logistic Regression Model
# ===================================================================

model = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# Class predictions
y_pred = model.predict(X_test_scaled)

# Probability predictions
y_proba = model.predict_proba(X_test_scaled)

# Probability of class 1
y_proba_class_1 = y_proba[:, 1]


# ===================================================================
# TASK 4: Classification Metrics Evaluation
# ===================================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_proba_class_1
)


print("=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(cm)

print(f"\nROC-AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ===================================================================
# TASK 5: Hyperparameter Regularization Experiment (C Parameter)
# ===================================================================

c_values = [0.01, 1.0, 100.0]

print("\n" + "=" * 60)
print("C PARAMETER EXPERIMENT")
print("=" * 60)

for c in c_values:

    model_c = LogisticRegression(
        C=c,
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )

    model_c.fit(
        X_train_scaled,
        y_train
    )

    # Predictions
    y_pred_c = model_c.predict(X_test_scaled)

    # Probability of class 1
    y_proba_c = model_c.predict_proba(X_test_scaled)[:, 1]

    # Accuracy
    accuracy_c = accuracy_score(
        y_test,
        y_pred_c
    )

    # ROC-AUC
    roc_auc_c = roc_auc_score(
        y_test,
        y_proba_c
    )

    # L2 norm of coefficients
    l2_norm = np.linalg.norm(
        model_c.coef_
    )

    print(
        f"C={c:<6} | "
        f"Accuracy={accuracy_c:.4f} | "
        f"ROC-AUC={roc_auc_c:.4f} | "
        f"L2 Norm={l2_norm:.4f}"
    )


# ===================================================================
# FINAL INTERPRETATION
# ===================================================================

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print("""
C controls the strength of regularization.

Small C:
    Stronger regularization
    Smaller coefficients
    Higher bias / potentially lower variance

Large C:
    Weaker regularization
    Larger coefficients
    Lower bias / potentially higher variance

Remember:

    Smaller C  -> Stronger regularization
    Larger C   -> Weaker regularization
""")