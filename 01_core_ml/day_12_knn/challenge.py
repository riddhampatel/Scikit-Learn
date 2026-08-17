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

# TODO: Load the Wine dataset using load_wine()
data = None
X = None
y = None

# TODO: Perform an 80/20 train/test split with random_state=42 and stratify=y
X_train, X_test, y_train, y_test = None, None, None, None


# ===================================================================
# TASK 2: Feature Scaling
# ===================================================================

# TODO: Instantiate StandardScaler
scaler = None

# TODO: Fit scaler ONLY on X_train and transform X_train
X_train_scaled = None

# TODO: Transform X_test using the fitted scaler
X_test_scaled = None


# ===================================================================
# TASK 3: Fit KNeighborsClassifier & Evaluate
# ===================================================================

# TODO: Instantiate KNeighborsClassifier with n_neighbors=5
model = None

# TODO: Fit the model on X_train_scaled and y_train


# TODO: Predict class labels for X_test_scaled
y_pred = None

# TODO: Calculate accuracy score on test set
accuracy = None

print("=" * 60)
print("k-NN CLASSIFIER RESULTS (SCALED)")
print("=" * 60)
print(f"Test Accuracy: {accuracy}")


# ===================================================================
# TASK 4: Feature Scaling Impact Comparison
# ===================================================================

# TODO: Instantiate and fit a KNeighborsClassifier(n_neighbors=5) on UNSCALED X_train
model_unscaled = None


# TODO: Predict on UNSCALED X_test
y_pred_unscaled = None

# TODO: Calculate accuracy score on UNSCALED test set
accuracy_unscaled = None

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
    # TODO: Instantiate KNeighborsClassifier with n_neighbors=k
    knn_k = None
    
    # TODO: Fit on X_train_scaled
    
    # TODO: Calculate training accuracy and test accuracy
    train_acc = None
    test_acc = None
    
    print(f"k={k:<3} | Train Accuracy: {train_acc} | Test Accuracy: {test_acc}")


# ===================================================================
# TASK 6: k-NN Regression (KNeighborsRegressor)
# ===================================================================

# Load Diabetes dataset
diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# TODO: Scale diabetes features using StandardScaler
scaler_r = None
X_train_r_scaled = None
X_test_r_scaled = None

# TODO: Instantiate KNeighborsRegressor with n_neighbors=5 and weights='distance'
knn_reg = None

# TODO: Fit regressor on scaled training data


# TODO: Predict on scaled test data
y_pred_r = None

# TODO: Calculate RMSE and R2 score
rmse = None
r2 = None

print("\n" + "=" * 60)
print("k-NN REGRESSOR RESULTS")
print("=" * 60)
print(f"RMSE: {rmse}")
print(f"R²:   {r2}")
