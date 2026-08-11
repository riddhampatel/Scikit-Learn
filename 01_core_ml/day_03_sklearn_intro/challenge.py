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
wine = load_wine()
X = wine.data
y = wine.target
print("X  .shape:", X.shape)
print("y.shape:", y.shape)
print("wine.target_names:", wine.target_names)





# ===================================================================
# TASK 2: Transformer Practice (StandardScaler)
# ===================================================================
# 1. Instantiate `StandardScaler` into variable `scaler`
# 2. Fit the scaler on feature matrix `X`
# 3. Transform `X` into `X_scaled`
# 4. Print the mean of feature 0 after scaling (should be close to 0)

# TODO: Create scaler, fit & transform X, and print the mean of X_scaled[:, 0]
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
print("Mean of feature 0:", X_scaled[:, 0].mean())


# ===================================================================
# TASK 3: Estimator & Predictor Workflow
# ===================================================================
# 1. Instantiate `KNeighborsClassifier` with `n_neighbors=5` into variable `model`
# 2. Fit the classifier using `X_scaled` and target vector `y`
# 3. Predict the class for sample #0 (`X_scaled[0:1]`) and print the predicted class vs actual class (`y[0]`)
# 4. Print the overall training accuracy using `model.score(X_scaled, y)`

# TODO: Instantiate model, fit on X_scaled and y, predict sample #0, and print score
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_scaled, y)
prediction = model.predict(X_scaled[0:1])

print("Predicted class:", prediction[0])
print("Actual class:", y[0])
accuracy = model.score(X_scaled, y)

print("Training Accuracy:", accuracy)