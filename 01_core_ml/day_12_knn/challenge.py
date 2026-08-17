"""
DAY 12 ACTIVE RECALL CHALLENGE: k-Nearest Neighbors (k-NN)
"""

import numpy as np

from sklearn.datasets import load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score


# ===================================================================
# TASK 1: Load Dataset & Train/Test Split
# ===================================================================

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ===================================================================
# TASK 2: Feature Scaling
# ===================================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===================================================================
# TASK 3: Fit KNeighborsClassifier & Evaluate
# ===================================================================

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 60)
print("k-NN CLASSIFIER RESULTS (SCALED)")
print("=" * 60)
print(f"Test Accuracy: {accuracy}")


# ===================================================================
# TASK 4: Feature Scaling Impact Comparison
# ===================================================================

model_unscaled = KNeighborsClassifier(n_neighbors=5)

model_unscaled.fit(X_train, y_train)

y_pred_unscaled = model_unscaled.predict(X_test)

accuracy_unscaled = accuracy_score(y_test, y_pred_unscaled)

print("\n" + "=" * 60)
print("FEATURE SCALING COMPARISON")
print("=" * 60)
print(f"Accuracy Without Scaling: {accuracy_unscaled}")
print(f"Accuracy With Scaling:    {accuracy}")


# ===================================================================
# TASK 5: Hyperparameter Experiment (Tuning k / n_neighbors)
# ===================================================================

k_values = [1, 3, 5, 9, 15, 25]

print("\n" + "=" * 60)
print("EXPERIMENTING WITH k (n_neighbors)")
print("=" * 60)

for k in k_values:

    knn_k = KNeighborsClassifier(n_neighbors=k)

    knn_k.fit(X_train_scaled, y_train)

    train_acc = accuracy_score(
        y_train,
        knn_k.predict(X_train_scaled)
    )

    test_acc = accuracy_score(
        y_test,
        knn_k.predict(X_test_scaled)
    )

    print(f"k={k:<3} | Train Accuracy: {train_acc} | Test Accuracy: {test_acc}")


# ===================================================================
# TASK 6: k-NN Regression (KNeighborsRegressor)
# ===================================================================

diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_r = StandardScaler()

X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)

knn_reg = KNeighborsRegressor(
    n_neighbors=5,
    weights="distance"
)

knn_reg.fit(X_train_r_scaled, y_train_r)

y_pred_r = knn_reg.predict(X_test_r_scaled)

rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2 = r2_score(y_test_r, y_pred_r)

print("\n" + "=" * 60)
print("k-NN REGRESSOR RESULTS")
print("=" * 60)
print(f"RMSE: {rmse}")
print(f"R²:   {r2}")