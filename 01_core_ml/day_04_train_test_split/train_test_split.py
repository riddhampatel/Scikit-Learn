"""
===================================================================
DAY 04: Train/Test Split & Data Leakage Prevention
===================================================================
Topics Covered:
1. Performing reproducible & stratified splits with train_test_split()
2. Checking shape & class distribution across splits
3. Correct Preprocessing Pipeline (Avoiding Data Leakage!)
4. Training on X_train and Evaluating on X_test
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# -----------------------------------------------------------------
# 1. LOAD DATASET & TRAIN/TEST SPLIT
# -----------------------------------------------------------------
print("=== 1. DATASET LOADING & STRATIFIED SPLIT ===")

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print(f"Full Dataset X shape: {X.shape}, y shape: {y.shape}")

# Perform 80% Train, 20% Test split with stratification & fixed random state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Testing set:  X_test  = {X_test.shape},  y_test  = {y_test.shape}")

# Verify Class Balance Preservation (Stratification)
orig_ratio = np.mean(y == 1)
train_ratio = np.mean(y_train == 1)
test_ratio = np.mean(y_test == 1)

print(f"\nClass 1 (Benign) Ratio in Full Dataset: {orig_ratio:.3f}")
print(f"Class 1 (Benign) Ratio in Train Set:    {train_ratio:.3f}")
print(f"Class 1 (Benign) Ratio in Test Set:     {test_ratio:.3f}")


# -----------------------------------------------------------------
# 2. PREPROCESSING WITHOUT DATA LEAKAGE
# -----------------------------------------------------------------
print("\n=== 2. PREPROCESSING (DATA LEAKAGE PREVENTION) ===")

# STEP A: Instantiate Scaler
scaler = StandardScaler()

# STEP B: Fit & Transform ONLY on X_train
X_train_scaled = scaler.fit_transform(X_train)

# STEP C: Transform X_test using parameters learned from X_train!
X_test_scaled = scaler.transform(X_test)

print("Learned Mean on Train set (first 3 features):", scaler.mean_[:3].round(3))
print("Mean of X_train_scaled feature 0:", np.mean(X_train_scaled[:, 0]).round(5))
print("Mean of X_test_scaled feature 0 (Not 0!):", np.mean(X_test_scaled[:, 0]).round(3))


# -----------------------------------------------------------------
# 3. MODEL TRAINING & UNSEEN EVALUATION
# -----------------------------------------------------------------
print("\n=== 3. ESTIMATOR EVALUATION ON TEST SET ===")

model = KNeighborsClassifier(n_neighbors=5)

# Fit model ONLY on training data
model.fit(X_train_scaled, y_train)

# Evaluate model performance on UNSEEN test data
train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)

print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy:     {test_acc * 100:.2f}%")
