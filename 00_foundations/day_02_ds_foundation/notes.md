# 📝 Day 02 — Data Science Foundation (NumPy, Pandas & ML Math)

## What is it?
The foundational data manipulation, linear algebra, and statistical operations required to prepare datasets and execute operations in Scikit-Learn.

## Core Concepts & Syntax

### 1. NumPy Operations
- **Array Shape & Reshaping**: `arr.reshape(-1, 1)` converts a 1D array of shape `(n,)` into a 2D column matrix of shape `(n, 1)`. Scikit-Learn's `X` matrix *must* always be 2D!
- **Dot Product / Matrix Multiplication**: `np.dot(X, w)` or `X @ w`. Represents linear model inference $y = X \cdot w + b$.

### 2. Pandas Essentials
- **loc vs iloc**:
  - `df.loc[condition, [cols]]`: Label-based indexing (e.g. `df.loc[df['age'] > 30, 'salary']`).
  - `df.iloc[row_idx, col_idx]`: Pure integer position-based indexing (e.g. `df.iloc[0:5, 0:2]`).
- **Missing Data Handling**:
  - Check missing: `df.isna().sum()`
  - Impute numeric: `df['col'] = df['col'].fillna(df['col'].median())`
  - Drop missing: `df.dropna()`

### 3. ML Math & Standardization Formula
- **Mean** ($\mu$): $\mu = \frac{1}{N} \sum x_i$ (`np.mean(x)`)
- **Standard Deviation** ($\sigma$): $\sigma = \sqrt{\frac{1}{N} \sum (x_i - \mu)^2}$ (`np.std(x)`)
- **Z-Score Standardization**:
  $$z = \frac{x - \mu}{\sigma}$$
  *(This is exactly what Scikit-Learn's `StandardScaler` computes!)*

---

## What I Understood
- Why single features must be reshaped with `.reshape(-1, 1)` before feeding to `model.fit(X, y)`: Scikit-Learn expects $X$ to be a 2D array of `(n_samples, n_features)`.

## What Confused Me
- The difference between `df.dropna()` (drops rows) vs `df.fillna()` (replaces `NaN` with values).

---

## 🎤 Top Interview Questions

1. **Why does Scikit-Learn require feature matrix $X$ to be 2-dimensional while target $y$ can be 1-dimensional?**
   - *Answer*: $X$ represents a matrix of samples and features `(n_samples, n_features)` even if there is only 1 feature. $y$ represents a target vector `(n_samples,)`.

2. **What is the mathematical difference between Standard Deviation and Variance?**
   - *Answer*: Variance is the average squared distance from the mean ($\sigma^2$). Standard Deviation ($\sigma$) is the square root of variance, returning metrics to the original feature units.

3. **Why should missing values be filled with the Median instead of the Mean in skewed datasets?**
   - *Answer*: The mean is heavily sensitive to extreme outliers, whereas the median represents the robust 50th percentile.

4. **What is Broadcasting in NumPy?**
   - *Answer*: Broadcasting allows NumPy to perform arithmetic operations on arrays of different shapes automatically without duplicating data in memory.
