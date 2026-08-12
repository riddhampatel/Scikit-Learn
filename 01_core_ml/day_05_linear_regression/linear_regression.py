"""
===================================================================
DAY 05: Linear Regression Baseline & OLS
===================================================================
Topics Covered:
1. Loading & Preparing Regression Dataset (Diabetes Dataset)
2. Fitting LinearRegression Baseline
3. Inspecting Learned Parameters: coef_ (weights) & intercept_ (bias)
4. Generating Predictions & Calculating Residuals (y - y_pred)
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# -----------------------------------------------------------------
# 1. LOAD DATASET & TRAIN/TEST SPLIT
# -----------------------------------------------------------------
print("=== 1. DATASET LOADING (DIABETES REGRESSION) ===")

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

print(f"Feature Matrix X shape: {X.shape} (442 patients, 10 physiological features)")
print(f"Target Vector y shape:  {y.shape} (Quantitative disease progression index)")
print(f"Feature Names:          {diabetes.feature_names}")

# Split into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features properly (no data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------------------------------------------
# 2. FIT LINEAR REGRESSION MODEL
# -----------------------------------------------------------------
print("\n=== 2. FITTING LINEAR REGRESSION ===")

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Inspect weights (w) and bias (b)
print(f"Intercept (b / bias):     {model.intercept_:.2f}")
print("Feature Coefficients (w / slopes):")
for feature_name, coef in zip(diabetes.feature_names, model.coef_):
    print(f"  - {feature_name:<10}: {coef:>7.2f}")


# -----------------------------------------------------------------
# 3. PREDICTIONS, RESIDUALS & SCORE
# -----------------------------------------------------------------
print("\n=== 3. PREDICTIONS & RESIDUALS ===")

# Predict target values on test set
y_pred = model.predict(X_test_scaled)

# Calculate Residuals: actual - predicted
residuals = y_test - y_pred

print("First 5 Actual y_test values:   ", y_test[:5].round(1))
print("First 5 Predicted y_pred values:", y_pred[:5].round(1))
print("First 5 Residual Errors (e):    ", residuals[:5].round(1))

# Model R^2 Score
r2_train = model.score(X_train_scaled, y_train)
r2_test = model.score(X_test_scaled, y_test)

print(f"\nTraining R² Score: {r2_train:.4f}")
print(f"Testing  R² Score: {r2_test:.4f}")
