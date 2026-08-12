"""
DAY 04 ACTIVE RECALL CHALLENGE: Train/Test Split & Data Leakage Prevention
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# ===================================================================
# TASK 1: Stratified Train/Test Split
# ===================================================================
# 1. Load the breast cancer dataset using `load_breast_cancer()`
# 2. Extract feature matrix `X` and target vector `y`
# 3. Split the data into 75% training and 25% testing using `train_test_split()`
#    - Set `test_size=0.25`
#    - Set `random_state=42`
#    - Set `stratify=y`
# 4. Print the shape of `X_train`, `X_test`, `y_train`, and `y_test`

# TODO: Load dataset and perform 75/25 stratified split






# ===================================================================
# TASK 2: Scaler Preprocessing Without Data Leakage
# ===================================================================
# 1. Instantiate `StandardScaler`
# 2. Fit AND transform `X_train` into `X_train_scaled` using `fit_transform()`
# 3. Transform `X_test` into `X_test_scaled` using ONLY `transform()`
# 4. Print the mean of feature 0 for both `X_train_scaled` and `X_test_scaled`

# TODO: Fit scaler on training set, transform train & test sets, and print feature 0 means






# ===================================================================
# TASK 3: Model Training & Unseen Evaluation
# ===================================================================
# 1. Instantiate `KNeighborsClassifier(n_neighbors=7)`
# 2. Train/fit the model using `X_train_scaled` and `y_train`
# 3. Compute and print training accuracy using `.score(X_train_scaled, y_train)`
# 4. Compute and print testing accuracy using `.score(X_test_scaled, y_test)`

# TODO: Fit classifier on training set, evaluate and print train and test scores
