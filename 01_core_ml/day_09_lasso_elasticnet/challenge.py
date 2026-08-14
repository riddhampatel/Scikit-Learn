"""
DAY 09 ACTIVE RECALL CHALLENGE: Lasso Regression & ElasticNet
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LassoCV,
    ElasticNetCV
)
from sklearn.metrics import r2_score, root_mean_squared_error


# ===================================================================
# TASK 1: Dataset Generation & Train/Test Split
# ===================================================================

# pyrefly: ignore [bad-unpacking]
X, y = make_regression(
    n_samples=150,
    n_features=30,
    n_informative=5,
    noise=12.0,
    coef=False,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ===================================================================
# TASK 2: Feature Scaling
# ===================================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===================================================================
# TASK 3: Train Lasso and ElasticNet Models
# ===================================================================

lasso_model = Lasso(
    alpha=1.0,
    random_state=42
)

lasso_model.fit(X_train_scaled, y_train)


elastic_model = ElasticNet(
    alpha=1.0,
    l1_ratio=0.5,
    random_state=42
)

elastic_model.fit(X_train_scaled, y_train)


ols_model = LinearRegression()

ols_model.fit(
    X_train_scaled,
    y_train
)


# Predictions

y_pred_lasso = lasso_model.predict(X_test_scaled)

y_pred_elastic = elastic_model.predict(X_test_scaled)

y_pred_ols = ols_model.predict(X_test_scaled)


# ===================================================================
# TASK 4: Feature Selection & Metric Evaluation
# ===================================================================

# -------------------------
# R² scores
# -------------------------

ols_r2 = r2_score(y_test, y_pred_ols)
lasso_r2 = r2_score(y_test, y_pred_lasso)
elastic_r2 = r2_score(y_test, y_pred_elastic)


# -------------------------
# RMSE scores
# -------------------------

ols_rmse = root_mean_squared_error(
    y_test,
    y_pred_ols
)

lasso_rmse = root_mean_squared_error(
    y_test,
    y_pred_lasso
)

elastic_rmse = root_mean_squared_error(
    y_test,
    y_pred_elastic
)


# -------------------------
# Active / non-zero features
# -------------------------

ols_active = np.sum(
    np.abs(ols_model.coef_) > 1e-4
)

lasso_active = np.sum(
    np.abs(lasso_model.coef_) > 1e-4
)

elastic_active = np.sum(
    np.abs(elastic_model.coef_) > 1e-4
)


# -------------------------
# Print results
# -------------------------

print("===== OLS vs Lasso vs ElasticNet =====")

print("\nTest R²:")
print(f"OLS:        {ols_r2:.4f}")
print(f"Lasso:      {lasso_r2:.4f}")
print(f"ElasticNet: {elastic_r2:.4f}")

print("\nTest RMSE:")
print(f"OLS:        {ols_rmse:.4f}")
print(f"Lasso:      {lasso_rmse:.4f}")
print(f"ElasticNet: {elastic_rmse:.4f}")

print("\nActive Features:")
print(f"OLS:        {ols_active}")
print(f"Lasso:      {lasso_active}")
print(f"ElasticNet: {elastic_active}")


# ===================================================================
# TASK 5: Hyperparameter Tuning via LassoCV & ElasticNetCV
# ===================================================================

# -------------------------
# LassoCV
# -------------------------

lasso_cv = LassoCV(
    alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
    cv=5,
    random_state=42
)

lasso_cv.fit(
    X_train_scaled,
    y_train
)


# -------------------------
# ElasticNetCV
# -------------------------

elastic_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.9],
    cv=5,
    random_state=42
)

elastic_cv.fit(
    X_train_scaled,
    y_train
)


# -------------------------
# Predictions
# -------------------------

y_pred_lasso_cv = lasso_cv.predict(X_test_scaled)

y_pred_elastic_cv = elastic_cv.predict(X_test_scaled)


# -------------------------
# Test R²
# -------------------------

lasso_cv_r2 = r2_score(
    y_test,
    y_pred_lasso_cv
)

elastic_cv_r2 = r2_score(
    y_test,
    y_pred_elastic_cv
)


# -------------------------
# Print tuned results
# -------------------------

print("\n===== LassoCV =====")

print(f"Best Alpha: {lasso_cv.alpha_:.6f}")
print(f"Test R²:    {lasso_cv_r2:.4f}")


print("\n===== ElasticNetCV =====")

print(f"Best Alpha:    {elastic_cv.alpha_:.6f}")
print(f"Best L1 Ratio: {elastic_cv.l1_ratio_:.2f}")
print(f"Test R²:       {elastic_cv_r2:.4f}")