"""
===================================================================
DAY 09: Lasso Regression (L1) & ElasticNet (L1 + L2)
===================================================================
Topics Covered:
1. Sparse Dataset Generation with Irrelevant Features
2. L1 Penalty & Automatic Feature Selection in Lasso
3. ElasticNet: Combining L1 and L2 Regularization
4. Comparing OLS, Ridge, Lasso, and ElasticNet
5. Hyperparameter Tuning using LassoCV and ElasticNetCV
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error


# -----------------------------------------------------------------
# 1. GENERATE SPARSE HIGH-DIMENSIONAL DATASET
# -----------------------------------------------------------------
print("=== 1. GENERATING HIGH-DIMENSIONAL SPARSE DATASET ===")

np.random.seed(42)

# 150 samples, 50 features, but ONLY 8 features are truly informative!
# pyrefly: ignore [bad-unpacking]
X, y = make_regression(
    n_samples=150,
    n_features=50,
    n_informative=8,
    noise=10.0,
    coef=False,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"Total features: {X.shape[1]}")
print(f"True informative features: 8 (42 features are pure noise)")
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")


# -----------------------------------------------------------------
# 2. FEATURE SCALING
# -----------------------------------------------------------------
print("\n=== 2. FEATURE SCALING ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features standardized (mean=0, std=1) to ensure equal penalty weighting.")


# -----------------------------------------------------------------
# 3. COMPARE OLS, RIDGE, LASSO, AND ELASTICNET
# -----------------------------------------------------------------
print("\n=== 3. MODEL COMPARISON: OLS vs RIDGE vs LASSO vs ELASTICNET ===")

models = {
    "OLS (Linear)": LinearRegression(),
    "Ridge (alpha=10)": Ridge(alpha=10.0, random_state=42),
    "Lasso (alpha=1.0)": Lasso(alpha=1.0, random_state=42, max_iter=10000),
    "ElasticNet (a=1.0, l1=0.5)": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)
}

print(f"{'Model Name':<28} {'Train R^2':>10} {'Test R^2':>10} {'Test RMSE':>12} {'Active Coefs':>14} {'||w||_1':>10}")
print("-" * 88)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)

    # Count non-zero coefficients (active features selected)
    non_zero_count = np.sum(np.abs(model.coef_) > 1e-4)
    l1_norm = np.linalg.norm(model.coef_, ord=1)

    print(f"{name:<28} {train_r2:>10.4f} {test_r2:>10.4f} {test_rmse:>12.4f} {non_zero_count:>14d}/50 {l1_norm:>10.2f}")


# -----------------------------------------------------------------
# 4. INSPECT FEATURE SELECTION IN LASSO vs RIDGE
# -----------------------------------------------------------------
print("\n=== 4. FEATURE SELECTION INSPECTION (FIRST 10 COEFFICIENTS) ===")

ridge_model = models["Ridge (alpha=10)"]
lasso_model = models["Lasso (alpha=1.0)"]

print(f"{'Feature Index':<16} {'Ridge Weight (L2)':>20} {'Lasso Weight (L1)':>20}")
print("-" * 60)

for i in range(10):
    r_w = ridge_model.coef_[i]
    l_w = lasso_model.coef_[i]
    status = "(ELIMINATED)" if abs(l_w) < 1e-4 else "(SELECTED)"
    print(f"Feature {i:<8} {r_w:>20.4f} {l_w:>20.4f} {status}")


# -----------------------------------------------------------------
# 5. AUTOMATIC TUNING WITH LASSOCV AND ELASTICNETCV
# -----------------------------------------------------------------
print("\n=== 5. AUTOMATIC TUNING WITH LASSOCV & ELASTICNETCV ===")

# LassoCV
lasso_cv = LassoCV(alphas=np.logspace(-3, 2, 50), cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

y_pred_lasso_cv = lasso_cv.predict(X_test_scaled)
lasso_cv_r2 = r2_score(y_test, y_pred_lasso_cv)
lasso_cv_rmse = root_mean_squared_error(y_test, y_pred_lasso_cv)
lasso_active = np.sum(np.abs(lasso_cv.coef_) > 1e-4)

print("--- LassoCV Results ---")
print(f"Optimal Alpha: {lasso_cv.alpha_:.4f}")
print(f"Test R^2:      {lasso_cv_r2:.4f}")
print(f"Test RMSE:     {lasso_cv_rmse:.4f}")
print(f"Active Features Selected: {lasso_active}/50")

# ElasticNetCV
elastic_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99],
    alphas=np.logspace(-3, 2, 30),
    cv=5,
    random_state=42,
    max_iter=10000
)
elastic_cv.fit(X_train_scaled, y_train)

y_pred_elastic_cv = elastic_cv.predict(X_test_scaled)
elastic_cv_r2 = r2_score(y_test, y_pred_elastic_cv)
elastic_cv_rmse = root_mean_squared_error(y_test, y_pred_elastic_cv)
elastic_active = np.sum(np.abs(elastic_cv.coef_) > 1e-4)

print("\n--- ElasticNetCV Results ---")
print(f"Optimal Alpha:    {elastic_cv.alpha_:.4f}")
print(f"Optimal L1 Ratio: {elastic_cv.l1_ratio_:.4f}")
print(f"Test R^2:         {elastic_cv_r2:.4f}")
print(f"Test RMSE:        {elastic_cv_rmse:.4f}")
print(f"Active Features Selected: {elastic_active}/50")
