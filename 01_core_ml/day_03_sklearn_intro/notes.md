# 📝 Day 03 — What is Scikit-learn & Estimator API

## What is it?
Scikit-Learn (sklearn) is the premier Python library for classical Machine Learning. Built on top of NumPy, SciPy, and Matplotlib, it offers a consistent, simple, and clean API design across all algorithms.

---

## Core Concepts & Unified API Design

Scikit-Learn's entire architecture is built around 3 primary object types:

### 1. Estimator
Any object that learns from data is an **Estimator**.
- **Method**: `.fit(X, y)`
- **Role**: Fits parameters to the dataset (e.g., calculates slope/intercept in Linear Regression, or mean/std in StandardScaler).
- **Rule**: `.fit()` always returns `self` (allows method chaining).

### 2. Transformer
An Estimator that transforms data (preprocessing, feature scaling, encoding).
- **Methods**: 
  - `.transform(X)`: Applies learned transformation parameters to new data.
  - `.fit_transform(X, y)`: Fits parameters AND transforms $X$ in a single optimized step.

### 3. Predictor
An Estimator that makes predictions on new data.
- **Methods**:
  - `.predict(X)`: Predicts target output vector $\hat{y}$ for input matrix $X$.
  - `.predict_proba(X)`: Returns class probabilities for classification.
  - `.score(X, y)`: Computes evaluation metric ($R^2$ score for regression, Accuracy for classification).

---

## The 5 Essential Scikit-Learn Data Conventions

1. **Feature Matrix ($X$)**:
   - MUST be a **2D array** of shape `(n_samples, n_features)`.
   - Rows = individual samples (e.g., flowers, houses).
   - Columns = features (e.g., sepal length, square footage).

2. **Target Vector ($y$)**:
   - Usually a **1D array** of shape `(n_samples,)`.
   - Contains label classes or continuous target values.

3. **Supervised vs Unsupervised**:
   - Supervised: `.fit(X, y)` requires both feature matrix $X$ and target vector $y$.
   - Unsupervised: `.fit(X)` requires only feature matrix $X$.

4. **Estimated Parameters**:
   - Learned attributes end with an **underscore `_`** (e.g., `model.coef_`, `scaler.mean_`, `scaler.scale_`).

5. **Data Leakage Prevention**:
   - Always `.fit()` ONLY on training data, then `.transform()` or `.predict()` on validation/test data!

---

## 🎤 Top Interview Questions

1. **What is the difference between `.fit()`, `.transform()`, and `.fit_transform()`?**
   - *Answer*: `.fit()` calculates and stores parameters from the input data. `.transform()` applies those saved parameters to transform the data. `.fit_transform()` combines both steps efficiently, but should only be used on training data to avoid data leakage.

2. **Why do attributes learned during `.fit()` end with an trailing underscore (e.g., `coef_`, `mean_`)?**
   - *Answer*: In Scikit-Learn's API convention, trailing underscores distinguish learned parameters created during training from user-defined hyper-parameters passed during instantiation (e.g. `n_neighbors`).

3. **What happens if you pass a 1D NumPy array as feature matrix $X$ into `.fit()`?**
   - *Answer*: Scikit-Learn raises a `ValueError` because $X$ must be 2D `(n_samples, n_features)`. You must reshape 1D arrays using `.reshape(-1, 1)`.

4. **What metric does `model.score(X, y)` compute by default?**
   - *Answer*: For Classifiers, `score()` computes **Accuracy** (ratio of correct predictions). For Regressors, `score()` computes **$R^2$ (coefficient of determination)**.
