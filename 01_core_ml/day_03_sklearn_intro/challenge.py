"""
DAY 03 ACTIVE RECALL CHALLENGE: Scikit-learn API & Dataset Workflow

"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# ===================================================================
# TASK 1: Dataset Exploration & Shapes
# ===================================================================
# 1. Load the Wine dataset using `load_wine()` into variable `wine`
# 2. Extract feature matrix `X` and target vector `y`
# 3. Print the shape of `X`, shape of `y`, and target class names

# TODO: Load dataset, extract X and y, and print their shapes & target names






# ===================================================================
# TASK 2: Transformer Practice (StandardScaler)
# ===================================================================
# 1. Instantiate `StandardScaler` into variable `scaler`
# 2. Fit the scaler on feature matrix `X`
# 3. Transform `X` into `X_scaled`
# 4. Print the mean of feature 0 after scaling (should be close to 0)

# TODO: Create scaler, fit & transform X, and print the mean of X_scaled[:, 0]






# ===================================================================
# TASK 3: Estimator & Predictor Workflow
# ===================================================================
# 1. Instantiate `KNeighborsClassifier` with `n_neighbors=5` into variable `model`
# 2. Fit the classifier using `X_scaled` and target vector `y`
# 3. Predict the class for sample #0 (`X_scaled[0:1]`) and print the predicted class vs actual class (`y[0]`)
# 4. Print the overall training accuracy using `model.score(X_scaled, y)`

# TODO: Instantiate model, fit on X_scaled and y, predict sample #0, and print score
