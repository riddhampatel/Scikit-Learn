# 📝 Day 01 — Python for Machine Learning

## What is it?
Core Python fundamentals required for building, customizing, and executing Machine Learning workflows in Scikit-Learn.

## Core Concepts & Syntax

### 1. Data Structures
- **List** (`[]`): Mutable, ordered collection (e.g. feature lists).
- **Tuple** (`()`): Immutable, ordered collection (e.g. shape `(n_samples, n_features)`).
- **Dictionary** (`{}`): Key-value mappings (e.g. model parameters & metrics). Use `.get(key, default)` to prevent `KeyError`.
- **Set** (`set()`): Unordered collection of unique values (e.g. unique class labels).

### 2. Comprehensions
- List: `[x**2 for x in data if x > 0]`
- Dict: `{k: v.strip() for k, v in raw_data.items()}`

### 3. OOP & The Scikit-Learn Pattern
All Scikit-Learn estimators implement:
- `__init__(**params)`: Store hyperparameters.
- `fit(X, y)`: Estimate parameters from data (saves attributes with trailing underscore `self.mean_`).
- `predict(X)` / `transform(X)`: Apply learned parameters to new data.

---

## What I Understood
- Why trailing underscores (e.g. `model.coef_`, `model.min_val_`) are used in Scikit-Learn: to represent attributes learned *during* `fit()`.
- How list and dictionary comprehensions make preprocessing pipelines clean and fast.

## What Confused Me
- Difference between `*args` (passes positional arguments as a tuple) and `**kwargs` (passes keyword arguments as a dictionary).

---

## 🎤 Top Interview Questions

1. **Why are dataset shapes in Scikit-Learn represented as Tuples instead of Lists?**
   - *Answer*: Tuples are immutable, preventing accidental modification of structural metadata (`n_samples, n_features`).

2. **What is the difference between `fit()`, `transform()`, and `fit_transform()`?**
   - *Answer*: `fit()` calculates parameters (mean, std, min, max) from training data; `transform()` applies those parameters; `fit_transform()` does both in one step efficiently.

3. **How does Scikit-Learn distinguish between initial hyperparameters and learned parameters?**
   - *Answer*: Initial parameters are set in `__init__()`. Learned parameters are created during `fit()` and end with a trailing underscore `_` (e.g. `coef_`, `classes_`).

4. **Why should you use `.get()` when querying dictionary keys in ML configurations?**
   - *Answer*: `.get('param', default)` prevents code crashes from missing keys when parsing user configs or experimental parameters.
