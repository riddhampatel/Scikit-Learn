"""
DAY 06 ACTIVE RECALL CHALLENGE: Regression Evaluation Metrics
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)


# ===================================================================
# TASK 1: Dataset Preparation & Model Fitting
# ===================================================================

# 1. Load California Housing dataset
housing = fetch_california_housing()

# 2. Extract first 1000 samples
X = housing.data[:1000]
y = housing.target[:1000]

# 3. Train/test split: 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Standard scaling without data leakage
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Fit Linear Regression and predict
model = LinearRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


# ===================================================================
# TASK 2: Sklearn Evaluation Metrics (MAE, MSE, RMSE, R²)
# ===================================================================

# MAE
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

# MSE
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# RMSE
rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)

# R²
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)


# ===================================================================
# TASK 3: Manual Adjusted R² Formula
# ===================================================================

# Number of samples
n = X_test.shape[0]

# Number of features
p = X_test.shape[1]

# Adjusted R² formula
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("Adjusted R²:", adj_r2)