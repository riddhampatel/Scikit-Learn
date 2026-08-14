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
# 1. Generate synthetic regression data using `make_regression()`:
#    - n_samples=120, n_features=20, n_informative=10, noise=15.0, random_state=42
# 2. Perform an 80/20 Train/Test split with `random_state=42`
#    Assign outputs to `X_train`, `X_test`, `y_train`, `y_test`

# TODO: Generate regression data and perform 80/20 train/test split








# ===================================================================
# TASK 2: Feature Scaling
# ===================================================================
# 1. Instantiate `StandardScaler()`
# 2. Fit and transform `X_train` into `X_train_scaled`
# 3. Transform `X_test` into `X_test_scaled` (without fitting!)

# TODO: Standard scale training and testing data cleanly








# ===================================================================
# TASK 3: Fit Baseline OLS vs Ridge Regression
# ===================================================================
# 1. Fit `LinearRegression()` on `X_train_scaled` and `y_train` -> `ols_model`
# 2. Fit `Ridge(alpha=10.0)` on `X_train_scaled` and `y_train` -> `ridge_model`
# 3. Compute test set predictions `y_pred_ols` and `y_pred_ridge`

# TODO: Train OLS and Ridge models and get predictions on test data








# ===================================================================
# TASK 4: Evaluate Metrics & L2 Norm of Coefficients
# ===================================================================
# 1. Compute and print Test R² score for both OLS and Ridge models
# 2. Compute and print Test RMSE score for both OLS and Ridge models
# 3. Compute L2 norm of coefficient vectors:
#    - `ols_l2 = np.linalg.norm(ols_model.coef_)`
#    - `ridge_l2 = np.linalg.norm(ridge_model.coef_)`
# 4. Print both L2 norm values to verify weight shrinkage in Ridge

# TODO: Calculate and print metrics (R2, RMSE) and L2 norm of weights for both models








# ===================================================================
# TASK 5: Hyperparameter Tuning via RidgeCV
# ===================================================================
# 1. Define an alpha search grid: `alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]`
# 2. Instantiate and fit `RidgeCV(alphas=alphas, cv=5)` on `X_train_scaled` and `y_train`
# 3. Print the optimal alpha selected (`ridge_cv.alpha_`)
# 4. Evaluate and print Test R² score of the optimal Ridge model

# TODO: Tune alpha using 5-fold RidgeCV and display best alpha and test score







