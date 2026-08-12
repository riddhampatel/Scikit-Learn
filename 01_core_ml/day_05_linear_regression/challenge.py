"""
DAY 05 ACTIVE RECALL CHALLENGE: Linear Regression Baseline
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# ===================================================================
# TASK 1: Data Preparation & Train/Test Split
# ===================================================================
# 1. Load the Diabetes dataset using `load_diabetes()` into `diabetes`
# 2. Extract feature matrix `X` and target vector `y`
# 3. Perform 80/20 train/test split with `random_state=42` into `X_train`, `X_test`, `y_train`, `y_test`
# 4. Standard scale `X_train` and `X_test` without data leakage into `X_train_scaled`, `X_test_scaled`

# TODO: Load dataset, split 80/20, and scale features cleanly

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scale = StandardScaler()  
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)



# ===================================================================
# TASK 2: Linear Regression Model Training & Inspection
# ===================================================================
# 1. Instantiate `LinearRegression()` into variable `model`
# 2. Fit the model on `X_train_scaled` and `y_train`
# 3. Print the fitted model `intercept_` (bias)
# 4. Print the fitted model `coef_` (feature weights array)

# TODO: Instantiate LinearRegression, fit on train set, and print intercept & coefficients
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)






# ===================================================================
# TASK 3: Predictions, Residuals & Model Evaluation
# ===================================================================
# 1. Predict target values on `X_test_scaled` into variable `y_pred`
# 2. Compute residual array `residuals = y_test - y_pred` and print mean residual
# 3. Compute and print Training R² score using `model.score(X_train_scaled, y_train)`
# 4. Compute and print Testing R² score using `model.score(X_test_scaled, y_test)`

# TODO: Predict on test set, compute residuals, and print train/test R^2 scores

y_pred = model.predict(X_test_scaled)

residuals = y_test - y_pred

print("Mean Residual:", np.mean(residuals))
train_r2 = model.score(X_train_scaled, y_train)

print("Training R²:", train_r2)

test_r2 = model.score(X_test_scaled, y_test)

print("Testing R²:", test_r2)