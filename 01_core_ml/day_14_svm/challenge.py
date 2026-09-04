"""
DAY 14 ACTIVE RECALL CHALLENGE: Support Vector Machines (SVM)
"""

import numpy as np

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score


# ===================================================================
# TASK 1: Load Dataset & Stratified Train/Test Split
# ===================================================================
# 1. Load the Breast Cancer dataset using `load_breast_cancer()`.
# 2. Extract feature matrix X and target labels y.
# 3. Split into train and test sets (test_size=0.2, stratify=y, random_state=42).

data = load_breast_cancer()

# TODO: Extract feature matrix X and target labels y
X = None
y = None

# TODO: Perform stratified train/test split
X_train, X_test, y_train, y_test = None, None, None, None


# ===================================================================
# TASK 2: Feature Scaling (Mandatory for SVM)
# ===================================================================
# 1. Initialize `StandardScaler()`.
# 2. Fit and transform the training data (`X_train`).
# 3. Transform the test data (`X_test`) using the SAME scaler.

# TODO: Initialize StandardScaler and transform train/test sets
scaler = None
X_train_scaled = None
X_test_scaled = None


# ===================================================================
# TASK 3: Fit Linear SVC & Evaluate Metrics
# ===================================================================
# 1. Initialize `SVC(kernel="linear", C=1.0, random_state=42)`.
# 2. Fit on `X_train_scaled` and `y_train`.
# 3. Predict class labels on `X_test_scaled`.
# 4. Compute Accuracy, F1-Score, and ROC-AUC (using `decision_function`).

# TODO: Initialize and fit Linear SVC model
linear_svc = None

# TODO: Predict on test set
y_pred_linear = None

# TODO: Compute decision function scores for ROC-AUC
decision_scores = None

# TODO: Calculate evaluation metrics
accuracy_linear = None
f1_linear = None
roc_auc_linear = None

print("=" * 60)
print("LINEAR SVC EVALUATION")
print("=" * 60)
print(f"Accuracy:  {accuracy_linear}")
print(f"F1-Score:  {f1_linear}")
print(f"ROC-AUC:   {roc_auc_linear}")


# ===================================================================
# TASK 4: Inspect Support Vectors
# ===================================================================
# 1. Inspect the total number of support vectors using `support_vectors_`.
# 2. Inspect the number of support vectors allocated per class using `n_support_`.

# TODO: Extract total count and per-class support vector counts
total_support_vectors = None
support_per_class = None

print("\n" + "=" * 60)
print("SUPPORT VECTOR INSPECTION")
print("=" * 60)
print(f"Total Support Vectors:      {total_support_vectors}")
print(f"Support Vectors Per Class:  {support_per_class}")


# ===================================================================
# TASK 5: Non-Linear RBF Kernel Hyperparameter Grid (C and gamma)
# ===================================================================
# 1. Define candidate values for C and gamma.
# 2. Iterate through combinations, training `SVC(kernel="rbf", C=c, gamma=g, random_state=42)`.
# 3. Evaluate and print test accuracy for each combination.

c_candidates = [0.1, 1.0, 10.0]
gamma_candidates = [0.001, 0.01, 0.1, 1.0]

print("\n" + "=" * 60)
print("RBF KERNEL GRID EXPERIMENT (C x gamma)")
print("=" * 60)

for c in c_candidates:
    for g in gamma_candidates:
        # TODO: Initialize SVC with RBF kernel, C=c, gamma=g
        rbf_model = None

        # TODO: Fit model on X_train_scaled and y_train

        # TODO: Evaluate test accuracy
        test_acc = None

        print(f"C: {c:<5} | gamma: {g:<6} | Test Accuracy: {test_acc}")


# ===================================================================
# TASK 6: Support Vector Regression (SVR with RBF Kernel)
# ===================================================================
# 1. Load diabetes dataset using `load_diabetes()`.
# 2. Split into train and test sets (test_size=0.2, random_state=42).
# 3. Scale features using `StandardScaler`.
# 4. Initialize and fit `SVR(kernel="rbf", C=100.0, epsilon=5.0)`.
# 5. Compute RMSE and R² score on test set.

diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target

# TODO: Train/test split for regression
X_train_r, X_test_r, y_train_r, y_test_r = None, None, None, None

# TODO: Fit scaler on X_train_r and transform both train and test sets
scaler_r = None
X_train_r_scaled = None
X_test_r_scaled = None

# TODO: Initialize and fit SVR
svr_model = None

# TODO: Generate predictions on test set
y_pred_svr = None

# TODO: Compute RMSE and R2 score
rmse_svr = None
r2_svr = None

print("\n" + "=" * 60)
print("SUPPORT VECTOR REGRESSION (SVR)")
print("=" * 60)
print(f"Test RMSE: {rmse_svr}")
print(f"Test R²:   {r2_svr}")
