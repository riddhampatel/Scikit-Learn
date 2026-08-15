"""
===================================================================
DAY 11: Classification Metrics
===================================================================
Topics Covered:
1. Accuracy, Precision, Recall, F1-Score
2. Confusion Matrix & ConfusionMatrixDisplay
3. Comprehensive classification_report
4. ROC Curve & ROC-AUC Score (roc_curve, roc_auc_score)
5. Decision Threshold Tuning (Precision-Recall Tradeoff)
"""

import numpy as np
from sklearn.datasets import make_classification
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
    roc_curve,
    roc_auc_score,
)


def main():
    print("=" * 60)
    print("DAY 11: CLASSIFICATION METRICS DEMONSTRATION")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. CREATE SYNTHETIC IMBALANCED DATASET (90% Neg, 10% Pos)
    # -------------------------------------------------------------
    print("\n1. Generating Synthetic Imbalanced Dataset (90% Class 0, 10% Class 1)...")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.90, 0.10],
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples (Class 1 count: {np.sum(y_train)})")
    print(f"Testing set:  {X_test.shape[0]} samples (Class 1 count: {np.sum(y_test)})")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------
    # 2. FIT LOGISTIC REGRESSION MODEL
    # -------------------------------------------------------------
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Standard predictions at threshold 0.5
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # -------------------------------------------------------------
    # 3. INDIVIDUAL METRICS EVALUATION
    # -------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n2. Default Predictions (Threshold = 0.5):")
    print("-" * 45)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # -------------------------------------------------------------
    # 4. CONFUSION MATRIX
    # -------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n3. Confusion Matrix Breakdown:")
    print("-" * 45)
    print(f"True Negatives  (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives  (TP): {tp}")

    # Manual verification of formulas
    manual_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    manual_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    manual_f1 = (
        2 * (manual_prec * manual_rec) / (manual_prec + manual_rec)
        if (manual_prec + manual_rec) > 0
        else 0
    )

    print(f"\nManual Precision check: {manual_prec:.4f}")
    print(f"Manual Recall check:    {manual_rec:.4f}")
    print(f"Manual F1-Score check:  {manual_f1:.4f}")

    # -------------------------------------------------------------
    # 5. CLASSIFICATION REPORT
    # -------------------------------------------------------------
    print("\n4. Full Classification Report:")
    print("-" * 55)
    print(classification_report(y_test, y_pred, target_names=["Majority (0)", "Minority (1)"]))

    # -------------------------------------------------------------
    # 6. ROC CURVE & ROC-AUC SCORE
    # -------------------------------------------------------------
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    print("5. ROC-AUC Evaluation:")
    print("-" * 45)
    print(f"ROC-AUC Score: {auc_score:.4f}")

    # Sample a few thresholds along the ROC curve
    print("\nROC Curve Threshold Samples:")
    print(f"{'Threshold':<12} {'FPR':<10} {'TPR (Recall)':<12}")
    print("-" * 36)
    indices = np.linspace(0, len(thresholds) - 1, num=5, dtype=int)
    for idx in indices:
        print(f"{thresholds[idx]:<12.4f} {fpr[idx]:<10.4f} {tpr[idx]:<12.4f}")

    # -------------------------------------------------------------
    # 7. THRESHOLD TUNING (PRECISION-RECALL TRADEOFF)
    # -------------------------------------------------------------
    print("\n6. Precision-Recall Tradeoff via Custom Threshold Tuning:")
    print("-" * 65)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
    print("-" * 65)

    custom_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]

    for thresh in custom_thresholds:
        # Convert continuous probabilities into binary predictions based on thresh
        custom_pred = (y_proba >= thresh).astype(int)

        p = precision_score(y_test, custom_pred, zero_division=0)
        r = recall_score(y_test, custom_pred, zero_division=0)
        f = f1_score(y_test, custom_pred, zero_division=0)

        c_matrix = confusion_matrix(y_test, custom_pred)
        _tn, _fp, _fn, _tp = c_matrix.ravel()

        print(
            f"{thresh:<12.2f} {p:<12.4f} {r:<12.4f} {f:<12.4f} "
            f"{_tp:<6} {_fp:<6} {_fn:<6}"
        )

    print("\nKey Takeaway:")
    print("  * Lower threshold (e.g., 0.10) catches more positive cases (High Recall), but introduces false alarms (Low Precision).")
    print("  * Higher threshold (e.g., 0.90) ensures high confidence predictions (High Precision), but misses real cases (Low Recall).")


if __name__ == "__main__":
    main()
