"""
===================================================================
DAY 03: What is Scikit-Learn? & The Estimator API Architecture
===================================================================
Topics Covered:
1. Loading & Exploring Scikit-Learn Toy Datasets (Iris Dataset)
2. Feature Matrix X vs Target Vector y Conventions
3. The Transformer API: fit(), transform(), fit_transform()
4. The Predictor/Estimator API: fit(), predict(), score()
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# -----------------------------------------------------------------
# 1. EXPLORING SCIKIT-LEARN TOY DATASETS
# -----------------------------------------------------------------
print("=== 1. DATASET EXPLORATION (IRIS) ===")

# Load Iris dataset as a Bunch object (dictionary-like)
iris = load_iris()

X = iris.data          # Feature matrix (2D NumPy array)
y = iris.target        # Target vector (1D NumPy array)

print(f"Feature Matrix X Shape: {X.shape}  -> (n_samples, n_features)")
print(f"Target Vector y Shape:  {y.shape}        -> (n_samples,)")
print(f"Feature Names:          {iris.feature_names}")
print(f"Target Names (Classes): {iris.target_names}")

print("\nFirst 3 Samples (X):\n", X[:3])
print("First 3 Target Labels (y):", y[:3])


# -----------------------------------------------------------------
# 2. TRANSFORMER API: DATA PREPROCESSING (StandardScaler)
# -----------------------------------------------------------------
print("\n=== 2. TRANSFORMER API (StandardScaler) ===")

# Step 1: Instantiate the Transformer object
scaler = StandardScaler()

# Step 2: Fit on feature matrix X (computes mean and standard deviation)
scaler.fit(X)

# Learned parameters end with trailing underscore '_'
print(f"Learned Feature Means (scaler.mean_): {scaler.mean_}")
print(f"Learned Feature Scale (scaler.scale_): {scaler.scale_}")

# Step 3: Transform X using learned parameters
X_scaled = scaler.transform(X)
print(f"Scaled X Sample [0]: {X_scaled[0].round(3)}")

# Shortcut: fit_transform() does both steps together
X_scaled_shortcut = scaler.fit_transform(X)
print("fit_transform matches transform:", np.allclose(X_scaled, X_scaled_shortcut))


# -----------------------------------------------------------------
# 3. PREDICTOR / ESTIMATOR API (KNeighborsClassifier)
# -----------------------------------------------------------------
print("\n=== 3. ESTIMATOR & PREDICTOR API (K-NN Classifier) ===")

# Step 1: Instantiate the Estimator model
model = KNeighborsClassifier(n_neighbors=3)

# Step 2: Train/Fit the model on data (Supervised learning needs X and y)
model.fit(X_scaled, y)
print("Model fitted successfully:", model)

# Step 3: Make predictions on new unseen sample data
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Single sample: shape (1, 4)
new_sample_scaled = scaler.transform(new_sample)

prediction = model.predict(new_sample_scaled)
predicted_class = iris.target_names[prediction[0]]
print(f"\nNew Sample: {new_sample[0]}")
print(f"Predicted Class Index: {prediction[0]} ({predicted_class})")

# Step 4: Evaluate model accuracy on data
accuracy = model.score(X_scaled, y)
print(f"\nModel Training Accuracy score(): {accuracy * 100:.2f}%")
