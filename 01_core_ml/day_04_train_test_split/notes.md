# 📝 Day 04 — Train/Test Split & Data Leakage

## What is it?
Train/Test split is the foundational practice of partitioning a dataset into separate subsets: one to train the model (**Training Set**) and one to evaluate its performance on unseen data (**Testing Set**). Avoiding **Data Leakage** during this process is critical for building trustworthy ML models.

---

## Core Concepts & Parameters

### `train_test_split()` API (`sklearn.model_selection`)
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)
```

1. **`test_size`**: Proportion of dataset allocated to test set (commonly `0.2` for 80/20 or `0.25` for 75/25).
2. **`random_state`**: Seed for pseudo-random number generator. Ensures **reproducibility** across runs.
3. **`shuffle`**: Default `True`. Randomizes data before splitting to avoid bias from sorted or clustered datasets.
4. **`stratify=y`**: **Crucial for classification!** Preserves the proportion of target class labels in both train and test splits (essential for imbalanced datasets).

---

## ⚠️ Data Leakage: The Ultimate ML Sin

### What is Data Leakage?
Data Leakage occurs when information from outside the training dataset (e.g., test set or future data) is inadvertently used to train or preprocess the model. It leads to overly optimistic performance during testing, but severe model failure in production!

### Correct Preprocessing Order (Golden Rule)

❌ **INCORRECT (Causes Data Leakage)**:
```python
# WRONG: Scaler learns mean & std from the entire dataset including test data!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
```

✅ **CORRECT (Prevent Data Leakage)**:
```python
# 1. Split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Preprocess Train set using fit_transform()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. Preprocess Test set using ONLY transform()
X_test_scaled = scaler.transform(X_test)
```

---

## 🎤 Top Interview Questions

1. **Why must we fit transformers (like `StandardScaler`) ONLY on training data?**
   - *Answer*: To prevent data leakage. The mean ($\mu$) and standard deviation ($\sigma$) must reflect *only* knowledge available at training time. Using test data to calculate scaling parameters exposes unseen test information to the model.

2. **What is Stratified Splitting (`stratify=y`), and why is it important?**
   - *Answer*: Stratified splitting ensures that the train and test sets maintain the exact same target class proportion as the original dataset. Without it, random splitting might omit rare minority classes from the test or train set.

3. **What is the difference between parameters passed to `train_test_split` vs `.fit()`?**
   - *Answer*: `train_test_split` is a helper function that splits data matrices. `.fit()` is an estimator method that computes parameters from a training matrix.

4. **Why do we use `random_state` in `train_test_split`?**
   - *Answer*: Setting `random_state` fixes the random seed, making data splits completely deterministic and reproducible for debugging, team collaboration, and peer review.
