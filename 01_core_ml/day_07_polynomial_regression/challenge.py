"""
DAY 07 ACTIVE RECALL CHALLENGE: Polynomial Regression & Overfitting
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error


# ===================================================================
# TASK 1: Generate Nonlinear Data & Train/Test Split
# ===================================================================
# 1. Set `np.random.seed(42)`
# 2. Generate 80 samples: `X = np.sort(np.random.uniform(-3, 3, 80)).reshape(-1, 1)`
# 3. Create target: `y = 2*X² - 3*X + 1 + noise` (noise: `np.random.normal(0, 1.5, size=80)`)
# 4. Split 75/25 with `random_state=0` into `X_train, X_test, y_train, y_test`

# TODO: Generate nonlinear synthetic data and perform train/test split


# ===================================================================
# TASK 2: PolynomialFeatures Exploration
# ===================================================================
# 1. Create `PolynomialFeatures(degree=3, include_bias=False)` as `poly`
# 2. Transform `X_train` into `X_train_poly` using `fit_transform`
# 3. Print the number of generated features using `.shape`
# 4. Print feature names using `poly.get_feature_names_out()`

# TODO: Apply PolynomialFeatures and inspect the expanded feature matrix


# ===================================================================
# TASK 3: Degree Comparison Pipeline (Underfitting → Overfitting)
# ===================================================================
# 1. Loop over degrees `[1, 2, 3, 7, 12]`
# 2. For each degree, build a `make_pipeline` with:
#    - `PolynomialFeatures(degree=d, include_bias=False)`
#    - `StandardScaler()`
#    - `LinearRegression()`
# 3. Fit on `X_train, y_train`
# 4. Compute and print Train R², Test R², Train RMSE, Test RMSE for each degree

# TODO: Build pipeline for each degree and compare train vs test performance


# ===================================================================
# TASK 4: Identify the Overfitting Degree
# ===================================================================
# 1. From your results above, identify which degree(s) show overfitting
# 2. Print which degree is the best fit (highest test R² without large train-test gap)
# 3. For the most overfit model, print its max |coefficient| to show coefficient explosion

# TODO: Analyze results and identify overfitting vs good fit
