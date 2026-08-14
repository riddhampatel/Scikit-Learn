"""
===================================================================
DAY 08: Ridge Regression (L2 Regularization)
===================================================================
Topics Covered:
1. Multicollinearity & Overfitting in Ordinary Least Squares (OLS)
2. L2 Regularization & Coefficient Shrinkage
3. Impact of Feature Scaling on Ridge Penalty
4. Comparing Ridge Performance Across Alpha Grid
5. Efficient Hyperparameter Selection using RidgeCV
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error


# -----------------------------------------------------------------
# 1. GENERATE HIGHLY COLLINEAR & POLYNOMIAL DATA
# -----------------------------------------------------------------
print("=== 1. GENERATING SYNTHETIC DATASET ===")

np.random.seed(42)

# Base sample: 100 rows, 1 feature
X_base = np.sort(np.random.uniform(-3, 3, 100)).reshape(-1, 1)

# Target generated from cubic polynomial with noise
y = 1.5 * (X_base.ravel() ** 3) - 2 * (X_base.ravel() ** 2) + 0.5 * X_base.ravel() + np.random.normal(0, 3, size=100)

# Expand to degree 7 polynomial features (causes multicollinearity + potential overfitting)
poly = PolynomialFeatures(degree=7, include_bias=False)
X = poly.fit_transform(X_base)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"Feature matrix shape (Polynomial Degree 7): {X.shape}")
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")


# -----------------------------------------------------------------
# 2. DEMONSTRATE WHY SCALING MATTERS
# -----------------------------------------------------------------
print("\n=== 2. IMPORTANCE OF FEATURE SCALING IN RIDGE ===")

# Unscaled feature ranges vary widely across powers (x, x^2, ..., x^7)
feature_stds = np.std(X_train, axis=0)
print("Standard deviations of polynomial features (unscaled):")
for i, std_val in enumerate(feature_stds, 1):
    print(f"  Feature x^{i}: std = {std_val:.4f}")

print("\nNotice: High-degree terms have vastly larger variances/scales.")
print("Without StandardScaler, Ridge penalizes smaller features disproportionately!")


# -----------------------------------------------------------------
# 3. OLS vs RIDGE ACROSS VARIOUS ALPHA VALUES
# -----------------------------------------------------------------
print("\n=== 3. COMPARISON: OLS LINEAR REGRESSION VS RIDGE ===")

alphas = [0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

print(f"{'Model / Alpha':<16} {'Train R^2':>10} {'Test R^2':>10} {'Train RMSE':>12} {'Test RMSE':>12} {'||w||_2':>10}")
print("-" * 75)

for a in alphas:
    if a == 0:
        # Alpha=0 is pure OLS Linear Regression
        model = make_pipeline(StandardScaler(), LinearRegression())
        label = "OLS (alpha=0)"
    else:
        model = make_pipeline(StandardScaler(), Ridge(alpha=a, random_state=42))
        label = f"Ridge (alpha={a})"

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)

    # Extract weights (excluding intercept)
    if a == 0:
        weights = model.named_steps["linearregression"].coef_
    else:
        weights = model.named_steps["ridge"].coef_

    l2_norm = np.linalg.norm(weights)

    print(f"{label:<16} {train_r2:>10.4f} {test_r2:>10.4f} {train_rmse:>12.4f} {test_rmse:>12.4f} {l2_norm:>10.2f}")


# -----------------------------------------------------------------
# 4. COEFFICIENT SHRINKAGE TRACKING
# -----------------------------------------------------------------
print("\n=== 4. COEFFICIENT SHRINKAGE DETAILS ===")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Fit pure OLS
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)

# Fit Ridge with alpha=10
ridge_10 = Ridge(alpha=10.0, random_state=42)
ridge_10.fit(X_train_scaled, y_train)

# Fit Ridge with alpha=1000
ridge_1000 = Ridge(alpha=1000.0, random_state=42)
ridge_1000.fit(X_train_scaled, y_train)

print(f"{'Feature':<10} {'OLS Coef':>14} {'Ridge(alpha=10)':>18} {'Ridge(alpha=1000)':>20}")
print("-" * 65)

for i in range(X_train.shape[1]):
    print(f"x^{i+1:<8} {ols.coef_[i]:>14.4f} {ridge_10.coef_[i]:>18.4f} {ridge_1000.coef_[i]:>20.4f}")


# -----------------------------------------------------------------
# 5. AUTOMATIC ALPHA TUNING WITH RIDGECV
# -----------------------------------------------------------------
print("\n=== 5. AUTOMATIC ALPHA SELECTION WITH RIDGECV ===")

alpha_grid = np.logspace(-3, 3, 50)  # 50 values from 0.001 to 1000

ridge_cv_pipe = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=alpha_grid, cv=5, scoring="neg_mean_squared_error")
)

ridge_cv_pipe.fit(X_train, y_train)

best_alpha = ridge_cv_pipe.named_steps["ridgecv"].alpha_
best_ridge_model = ridge_cv_pipe

y_train_pred_cv = best_ridge_model.predict(X_train)
y_test_pred_cv = best_ridge_model.predict(X_test)

best_train_r2 = r2_score(y_train, y_train_pred_cv)
best_test_r2 = r2_score(y_test, y_test_pred_cv)
best_test_rmse = root_mean_squared_error(y_test, y_test_pred_cv)

print(f"Optimal Alpha found by RidgeCV: {best_alpha:.4f}")
print(f"Train R^2 : {best_train_r2:.4f}")
print(f"Test R^2  : {best_test_r2:.4f}")
print(f"Test RMSE : {best_test_rmse:.4f}")
