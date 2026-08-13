"""
===================================================================
DAY 06: Regression Evaluation Metrics (MAE, MSE, RMSE, R² & Adj R²)
===================================================================
Topics Covered:
1. Fitting a Linear Regression Model on Housing Data
2. Computing MAE, MSE, RMSE, and R² using sklearn.metrics
3. Implementing Custom Adjusted R² Formula
4. Comparing Metrics Across Train and Test Sets
"""

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
import numpy as np

# -----------------------------------------------------------------
# 1. PREPARE DATASET & TRAIN MODEL
# -----------------------------------------------------------------
print("=== 1. DATASET LOADING & MODEL TRAINING ===")

housing = fetch_california_housing()
X, y = housing.data[:2000], housing.target[:2000]  # Subset for fast computation

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features without leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Linear Regression baseline
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)


# -----------------------------------------------------------------
# 2. COMPUTE REGRESSION EVALUATION METRICS
# -----------------------------------------------------------------
print("\n=== 2. SKLEARN METRICS EVALUATION (TEST SET) ===")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE):     ${mae * 100000:.2f} ({mae:.4f})")
print(f"Mean Squared Error (MSE):      {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): ${rmse * 100000:.2f} ({rmse:.4f})")
print(f"R² Score (R-Squared):           {r2:.4f}")


# -----------------------------------------------------------------
# 3. CALCULATE ADJUSTED R² SCORE
# -----------------------------------------------------------------
print("\n=== 3. ADJUSTED R² COMPUTATION ===")

n = X_test.shape[0]  # Number of samples in test set
p = X_test.shape[1]  # Number of features

adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print(f"Test Set Samples (n): {n}, Features (p): {p}")
print(f"Standard R² Score: {r2:.4f}")
print(f"Adjusted R² Score: {adj_r2:.4f}")
