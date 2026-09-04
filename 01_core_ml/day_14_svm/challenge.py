"""
DAY 14 ACTIVE RECALL CHALLENGE: Support Vector Machines (SVM)
"""

import numpy as np

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score
)


# ===================================================================
# TASK 1: Load Dataset & Stratified Train/Test Split
# ===================================================================

data = load_breast_cancer()

# Extract feature matrix X and target labels y
X = data.data
y = data.target

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ===================================================================
# TASK 2: Feature Scaling (Mandatory for SVM)
# ===================================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===================================================================
# TASK 3: Fit Linear SVC & Evaluate Metrics
# ===================================================================

linear_svc = SVC(
    kernel="linear",
    C=1.0,
    random_state=42
)

linear_svc.fit(X_train_scaled, y_train)

# Predict class labels
y_pred_linear = linear_svc.predict(X_test_scaled)

# Decision function scores
decision_scores = linear_svc.decision_function(X_test_scaled)

# Evaluation metrics
accuracy_linear = accuracy_score(y_test, y_pred_linear)

f1_linear = f1_score(
    y_test,
    y_pred_linear
)

roc_auc_linear = roc_auc_score(
    y_test,
    decision_scores
)


print("=" * 60)
print("LINEAR SVC EVALUATION")
print("=" * 60)
print(f"Accuracy:  {accuracy_linear}")
print(f"F1-Score:  {f1_linear}")
print(f"ROC-AUC:   {roc_auc_linear}")


# ===================================================================
# TASK 4: Inspect Support Vectors
# ===================================================================

# Total number of support vectors
total_support_vectors = len(linear_svc.support_vectors_)

# Number of support vectors per class
support_per_class = linear_svc.n_support_


print("\n" + "=" * 60)
print("SUPPORT VECTOR INSPECTION")
print("=" * 60)
print(f"Total Support Vectors:      {total_support_vectors}")
print(f"Support Vectors Per Class:  {support_per_class}")


# ===================================================================
# TASK 5: Non-Linear RBF Kernel Hyperparameter Grid (C and gamma)
# ===================================================================

c_candidates = [0.1, 1.0, 10.0]
gamma_candidates = [0.001, 0.01, 0.1, 1.0]

print("\n" + "=" * 60)
print("RBF KERNEL GRID EXPERIMENT (C x gamma)")
print("=" * 60)

for c in c_candidates:
    for g in gamma_candidates:

        # Initialize RBF SVC
        rbf_model = SVC(
            kernel="rbf",
            C=c,
            gamma=g,
            random_state=42
        )

        # Fit model
        rbf_model.fit(
            X_train_scaled,
            y_train
        )

        # Predict
        y_pred_rbf = rbf_model.predict(
            X_test_scaled
        )

        # Evaluate accuracy
        test_acc = accuracy_score(
            y_test,
            y_pred_rbf
        )

        print(
            f"C: {c:<5} | gamma: {g:<6} | "
            f"Test Accuracy: {test_acc}"
        )


# ===================================================================
# TASK 6: Support Vector Regression (SVR with RBF Kernel)
# ===================================================================

diabetes = load_diabetes()

X_reg = diabetes.data
y_reg = diabetes.target

# Train/test split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler_r = StandardScaler()

X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)

# Initialize SVR
svr_model = SVR(
    kernel="rbf",
    C=100.0,
    epsilon=5.0
)

# Fit SVR
svr_model.fit(
    X_train_r_scaled,
    y_train_r
)

# Predictions
y_pred_svr = svr_model.predict(
    X_test_r_scaled
)

# RMSE
rmse_svr = np.sqrt(
    mean_squared_error(
        y_test_r,
        y_pred_svr
    )
)

# R²
r2_svr = r2_score(
    y_test_r,
    y_pred_svr
)


print("\n" + "=" * 60)
print("SUPPORT VECTOR REGRESSION (SVR)")
print("=" * 60)
print(f"Test RMSE: {rmse_svr}")
print(f"Test R²:   {r2_svr}")