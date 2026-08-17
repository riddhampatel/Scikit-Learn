"""
DAY 12 TUTORIAL: k-Nearest Neighbors (k-NN) Classification & Regression
"""

import sys
import numpy as np
from sklearn.datasets import load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score


def p(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def main():
    p("=" * 70)
    p("DAY 12: k-NEAREST NEIGHBORS (k-NN) TUTORIAL & EXPERIMENTS")
    p("=" * 70)

    # ------------------------------------------------------------------
    # 1. DEMONSTRATION: Why Feature Scaling is Mandatory for k-NN
    # ------------------------------------------------------------------
    p("\n1. IMPACT OF FEATURE SCALING ON CLASSIFICATION")
    p("-" * 50)

    # Load Wine dataset (features have drastically different scales: proline ~ 1000, ash ~ 2)
    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target

    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
        X_wine, y_wine, test_size=0.3, random_state=42, stratify=y_wine
    )

    # Model WITHOUT Feature Scaling
    knn_unscaled = KNeighborsClassifier(n_neighbors=5)
    knn_unscaled.fit(X_train_w, y_train_w)
    acc_unscaled = accuracy_score(y_test_w, knn_unscaled.predict(X_test_w))

    # Model WITH Feature Scaling
    scaler = StandardScaler()
    X_train_w_scaled = scaler.fit_transform(X_train_w)
    X_test_w_scaled = scaler.transform(X_test_w)

    knn_scaled = KNeighborsClassifier(n_neighbors=5)
    knn_scaled.fit(X_train_w_scaled, y_train_w)
    acc_scaled = accuracy_score(y_test_w, knn_scaled.predict(X_test_w_scaled))

    p(f"Accuracy WITHOUT Scaling: {acc_unscaled:.4f}")
    p(f"Accuracy WITH Scaling:    {acc_scaled:.4f}")
    p(f"Improvement from Scaling: +{(acc_scaled - acc_unscaled)*100:.2f}%")

    # ------------------------------------------------------------------
    # 2. TUNING K (n_neighbors) - Overfitting vs Underfitting
    # ------------------------------------------------------------------
    p("\n2. TUNING HYPERPARAMETER k (n_neighbors)")
    p("-" * 50)
    p(f"{'k Value':<10} | {'Train Accuracy':<15} | {'Test Accuracy':<15} | {'Status'}")
    p("-" * 55)

    k_values = range(1, 21, 2)  # Odd k values from 1 to 19
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_w_scaled, y_train_w)

        train_acc = accuracy_score(y_train_w, model.predict(X_train_w_scaled))
        test_acc = accuracy_score(y_test_w, model.predict(X_test_w_scaled))

        if k == 1:
            status = "High Variance (Overfitting)"
        elif k >= 15:
            status = "Higher Bias"
        else:
            status = "Balanced"

        p(f"k = {k:<6} | {train_acc:<15.4f} | {test_acc:<15.4f} | {status}")

    # ------------------------------------------------------------------
    # 3. WEIGHTED VOTING: 'uniform' vs 'distance'
    # ------------------------------------------------------------------
    p("\n3. WEIGHTING SCHEMES: 'uniform' vs 'distance'")
    p("-" * 50)

    knn_uniform = KNeighborsClassifier(n_neighbors=9, weights='uniform')
    knn_uniform.fit(X_train_w_scaled, y_train_w)
    acc_uniform = accuracy_score(y_test_w, knn_uniform.predict(X_test_w_scaled))

    knn_distance = KNeighborsClassifier(n_neighbors=9, weights='distance')
    knn_distance.fit(X_train_w_scaled, y_train_w)
    acc_distance = accuracy_score(y_test_w, knn_distance.predict(X_test_w_scaled))

    p(f"Accuracy with Uniform Weights:  {acc_uniform:.4f}")
    p(f"Accuracy with Distance Weights: {acc_distance:.4f}")

    # ------------------------------------------------------------------
    # 4. REGRESSION DEMONSTRATION: KNeighborsRegressor
    # ------------------------------------------------------------------
    p("\n4. k-NN REGRESSION (KNeighborsRegressor)")
    p("-" * 50)

    # Load Diabetes dataset
    diabetes = load_diabetes()
    X_d, y_d = diabetes.data, diabetes.target

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_d, y_d, test_size=0.2, random_state=42
    )

    scaler_reg = StandardScaler()
    X_train_d_scaled = scaler_reg.fit_transform(X_train_d)
    X_test_d_scaled = scaler_reg.transform(X_test_d)

    knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn_reg.fit(X_train_d_scaled, y_train_d)

    y_pred_reg = knn_reg.predict(X_test_d_scaled)
    rmse = np.sqrt(mean_squared_error(y_test_d, y_pred_reg))
    r2 = r2_score(y_test_d, y_pred_reg)

    p(f"KNeighborsRegressor RMSE: {rmse:.4f}")
    p(f"KNeighborsRegressor R²:   {r2:.4f}")

    p("\n" + "=" * 70)
    p("TUTORIAL COMPLETE!")
    p("=" * 70)


if __name__ == "__main__":
    main()
