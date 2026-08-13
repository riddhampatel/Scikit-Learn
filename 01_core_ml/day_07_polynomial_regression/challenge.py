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

# 1. Set random seed
np.random.seed(42)

# 2. Generate 80 samples
X = np.sort(
    np.random.uniform(-3, 3, 80)
).reshape(-1, 1)

# 3. Create nonlinear target with noise
noise = np.random.normal(0, 1.5, size=80)

y = 2 * (X.ravel()**2) - 3 * X.ravel() + 1 + noise

# Convert y from (80, 1) to (80,)
y = y.ravel()

# 4. Train/test split: 75/25
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0
)


# ===================================================================
# TASK 2: PolynomialFeatures Exploration
# ===================================================================

# 1. Create PolynomialFeatures
poly = PolynomialFeatures(
    degree=3,
    include_bias=False
)

# 2. Transform training data
X_train_poly = poly.fit_transform(X_train)

# 3. Print shape
print("Polynomial feature matrix shape:", X_train_poly.shape)

# 4. Print feature names
print("Feature names:")
print(poly.get_feature_names_out())


# ===================================================================
# TASK 3: Degree Comparison Pipeline
# ===================================================================

degrees = [1, 2, 3, 7, 12]

results = {}

for d in degrees:

    # Build pipeline
    model = make_pipeline(
        PolynomialFeatures(
            degree=d,
            include_bias=False
        ),
        StandardScaler(),
        LinearRegression()
    )

    # Fit model
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_rmse = root_mean_squared_error(
        y_train,
        y_train_pred
    )

    test_rmse = root_mean_squared_error(
        y_test,
        y_test_pred
    )

    # Store results
    results[d] = {
        "model": model,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse
    }

    # Print results
    print(f"\nDegree {d}")
    print("-" * 30)
    print(f"Train R²  : {train_r2:.4f}")
    print(f"Test R²   : {test_r2:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE : {test_rmse:.4f}")


# ===================================================================
# TASK 4: Identify the Overfitting Degree
# ===================================================================

# Find degree with highest test R²
best_degree = max(
    results,
    key=lambda d: results[d]["test_r2"]
)

print("\n" + "=" * 50)
print("MODEL ANALYSIS")
print("=" * 50)

print(f"Best degree based on Test R²: {best_degree}")

# Find overfitting using train-test R² gap
for d in degrees:

    train_r2 = results[d]["train_r2"]
    test_r2 = results[d]["test_r2"]

    gap = train_r2 - test_r2

    print(
        f"Degree {d}: "
        f"Train-Test R² Gap = {gap:.4f}"
    )


# Identify most overfit model
most_overfit_degree = max(
    degrees,
    key=lambda d: results[d]["train_r2"] - results[d]["test_r2"]
)

print(
    f"\nMost overfit degree: {most_overfit_degree}"
)

# Get the trained pipeline
overfit_model = results[most_overfit_degree]["model"]

# Extract LinearRegression step
linear_model = overfit_model.named_steps["linearregression"]

# Maximum absolute coefficient
max_coefficient = np.max(
    np.abs(linear_model.coef_)
)

print(
    f"Maximum |coefficient|: {max_coefficient:.4f}"
)