"""
Day 10 — Logistic Regression (Binary Classification)
Practice Script: Breast Cancer Classification using Scikit-Learn
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)


def main():
    print("==================================================")
    print("   DAY 10: LOGISTIC REGRESSION PRACTICE SCRIPT   ")
    print("==================================================\n")

    # 1. Load Dataset
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    feature_names = cancer.feature_names
    target_names = cancer.target_names

    print(f"Dataset Shape: X = {X.shape}, y = {y.shape}")
    print(f"Target Classes: 0 -> {target_names[0]} (Malignant), 1 -> {target_names[1]} (Benign)")
    print(f"Class Distribution: {np.bincount(y)} (0: Malignant, 1: Benign)\n")

    # 2. Train / Test Split (Stratified to maintain class ratios)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}\n")

    # 3. Feature Scaling (Mandatory for Logistic Regression gradient solvers & regularization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Model Training (Baseline Logistic Regression with default C=1.0)
    model = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    print("Model fitted successfully!\n")

    # 5. Predictions: Labels & Probabilities
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    print("Sample Predictions (First 5 samples):")
    print(f"  True Labels:        {y_test[:5]}")
    print(f"  Predicted Labels:   {y_pred[:5]}")
    print(f"  Predicted Probas [P(y=0), P(y=1)]:\n{y_proba[:5]}\n")

    # 6. Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    cm = confusion_matrix(y_test, y_pred)

    print("==================================================")
    print("            MODEL EVALUATION METRICS              ")
    print("==================================================")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"ROC-AUC:   {roc_auc:.4f}\n")

    print("Confusion Matrix:")
    print(f"  [[TN: {cm[0, 0]:2d}  FP: {cm[0, 1]:2d}]")
    print(f"   [FN: {cm[1, 0]:2d}  TP: {cm[1, 1]:2d}]]\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 7. Model Coefficient Inspection
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", key=abs, ascending=False)

    print("==================================================")
    print("   TOP 5 INFLUENTIAL FEATURES (BY ABSOLUTE WEIGHT) ")
    print("==================================================")
    print(coef_df.head(5).to_string(index=False))
    print(f"\nIntercept: {model.intercept_[0]:.4f}\n")

    # 8. Regularization Hyperparameter Comparison (C parameter)
    print("==================================================")
    print("    EFFECT OF REGULARIZATION STRENGTH (C)        ")
    print("==================================================")
    print(f"{'C Value':<10} | {'Train Acc':<10} | {'Test Acc':<10} | {'ROC-AUC':<10} | {'Coeff L2 Norm':<15}")
    print("-" * 65)

    for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        m = LogisticRegression(C=c, solver='lbfgs', max_iter=1000, random_state=42)
        m.fit(X_train_scaled, y_train)

        tr_acc = accuracy_score(y_train, m.predict(X_train_scaled))
        te_acc = accuracy_score(y_test, m.predict(X_test_scaled))
        auc = roc_auc_score(y_test, m.predict_proba(X_test_scaled)[:, 1])
        l2_norm = np.linalg.norm(m.coef_)

        print(f"{c:<10.3f} | {tr_acc:<10.4f} | {te_acc:<10.4f} | {auc:<10.4f} | {l2_norm:<15.4f}")


if __name__ == "__main__":
    main()
