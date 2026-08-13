"""
===================================================================
DAY 07: Polynomial Regression & Overfitting
===================================================================
Topics Covered:
1. PolynomialFeatures Transformer (Feature Expansion)
2. Building a Polynomial Regression Pipeline
3. Comparing Degree 1 vs 2 vs 5 vs 10 (Bias-Variance Tradeoff)
4. Detecting Overfitting via Train/Test Metric Gap
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error


# -----------------------------------------------------------------
# 1. GENERATE SYNTHETIC NONLINEAR DATA
# -----------------------------------------------------------------
print("=== 1. GENERATING SYNTHETIC NONLINEAR DATA ===")

np.random.seed(42)
X = np.sort(np.random.uniform(-3, 3, 100)).reshape(-1, 1)
# True function: y = 0.5x³ - 2x² + x + 3 + noise
y = 0.5 * X.ravel()**3 - 2 * X.ravel()**2 + X.ravel() + 3
y += np.random.normal(0, 2, size=y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")
print(f"True generating function: y = 0.5x^3 - 2x^2 + x + 3 + noise")


# -----------------------------------------------------------------
# 2. POLYNOMIALFEATURES TRANSFORMER DEMO
# -----------------------------------------------------------------
print("\n=== 2. POLYNOMIALFEATURES TRANSFORMER ===")

sample = np.array([[2, 3]])  # Single sample with 2 features
poly = PolynomialFeatures(degree=2, include_bias=False)
expanded = poly.fit_transform(sample)

print(f"Original features:   {sample} -> shape {sample.shape}")
print(f"Expanded features:   {expanded} -> shape {expanded.shape}")
print(f"Feature names:       {poly.get_feature_names_out()}")


# -----------------------------------------------------------------
# 3. FIT POLYNOMIAL MODELS AT VARYING DEGREES
# -----------------------------------------------------------------
print("\n=== 3. POLYNOMIAL DEGREE COMPARISON ===")
print(f"{'Degree':<8} {'Train R²':>10} {'Test R²':>10} {'Train RMSE':>12} {'Test RMSE':>12} {'Status':>12}")
print("-" * 66)

degrees = [1, 2, 3, 5, 10]

for d in degrees:
    # Pipeline: PolynomialFeatures → StandardScaler → LinearRegression
    pipe = make_pipeline(
        PolynomialFeatures(degree=d, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )
    pipe.fit(X_train, y_train)

    # Predict
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)

    # Diagnose fit status
    if test_r2 < 0.5 and train_r2 < 0.5:
        status = "UNDERFIT"
    elif train_r2 - test_r2 > 0.2 or test_r2 < 0:
        status = "OVERFIT"
    else:
        status = "GOOD FIT"

    print(f"{d:<8} {train_r2:>10.4f} {test_r2:>10.4f} {train_rmse:>12.4f} {test_rmse:>12.4f} {status:>12}")


# -----------------------------------------------------------------
# 4. OVERFITTING DEEP DIVE: DEGREE 10 vs DEGREE 3
# -----------------------------------------------------------------
print("\n=== 4. OVERFITTING DEEP DIVE ===")

# Degree 3 — true function is cubic, so this should generalize well
pipe_3 = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),
    LinearRegression()
)
pipe_3.fit(X_train, y_train)

# Degree 10  - too flexible, memorizes noise
pipe_10 = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    StandardScaler(),
    LinearRegression()
)
pipe_10.fit(X_train, y_train)

# Compare coefficient magnitudes (symptom of overfitting)
coefs_3 = pipe_3.named_steps["linearregression"].coef_
coefs_10 = pipe_10.named_steps["linearregression"].coef_

print(f"Degree 3  - Number of features: {len(coefs_3)}")
print(f"Degree 3  - Max |coefficient|:  {np.max(np.abs(coefs_3)):.2f}")
print(f"Degree 10  - Number of features: {len(coefs_10)}")
print(f"Degree 10  - Max |coefficient|:  {np.max(np.abs(coefs_10)):.2f}")
print("\n-> Exploding coefficient magnitudes are a classic symptom of overfitting!")
