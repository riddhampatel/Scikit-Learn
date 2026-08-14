"""
DAY 08 ACTIVE RECALL CHALLENGE: Ridge Regression (L2 Regularization)
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error


# ===================================================================
# TASK 1: Dataset Generation & Train/Test Split
# ===================================================================

X, y = make_regression(
    n_samples=120,
    n_features=20,
    n_informative=10,
    noise=15.0,
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
# TASK 3: Fit Baseline OLS vs Ridge Regression
# ===================================================================

ols_model = LinearRegression()
ols_model.fit(X_train_scaled, y_train)

ridge_model = Ridge(alpha=10.0)
ridge_model.fit(X_train_scaled, y_train)

y_pred_ols = ols_model.predict(X_test_scaled)
y_pred_ridge = ridge_model.predict(X_test_scaled)


# ===================================================================
# TASK 4: Evaluate Metrics & L2 Norm of Coefficients
# ===================================================================

ols_r2 = r2_score(y_test, y_pred_ols)
ridge_r2 = r2_score(y_test, y_pred_ridge)

ols_rmse = root_mean_squared_error(y_test, y_pred_ols)
ridge_rmse = root_mean_squared_error(y_test, y_pred_ridge)

ols_l2 = np.linalg.norm(ols_model.coef_)
ridge_l2 = np.linalg.norm(ridge_model.coef_)

print("===== OLS vs Ridge =====")

print(f"OLS Test R^2:       {ols_r2:.4f}")
print(f"Ridge Test R^2:     {ridge_r2:.4f}")

print(f"OLS Test RMSE:     {ols_rmse:.4f}")
print(f"Ridge Test RMSE:   {ridge_rmse:.4f}")

print(f"OLS Coeff L2 Norm:   {ols_l2:.4f}")
print(f"Ridge Coeff L2 Norm: {ridge_l2:.4f}")


# ===================================================================
# TASK 5: Hyperparameter Tuning via RidgeCV
# ===================================================================

alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

ridge_cv = RidgeCV(
    alphas=alphas,
    cv=5
)

ridge_cv.fit(X_train_scaled, y_train)

y_pred_ridge_cv = ridge_cv.predict(X_test_scaled)

ridge_cv_r2 = r2_score(y_test, y_pred_ridge_cv)

print("\n===== RidgeCV =====")

print(f"Best Alpha:       {ridge_cv.alpha_}")
print(f"RidgeCV Test R^2:  {ridge_cv_r2:.4f}")