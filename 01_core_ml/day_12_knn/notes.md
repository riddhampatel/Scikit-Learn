# 📝 Day 12 — k-Nearest Neighbors (k-NN)

## What is k-Nearest Neighbors?
**k-Nearest Neighbors (k-NN)** is one of the simplest and most intuitive **supervised learning algorithms**. It can be used for both **classification** and **regression** tasks.

Unlike parametric models (like Linear or Logistic Regression) that learn explicit weights $\mathbf{w}$ during a training phase, k-NN is an **instance-based** (or **lazy learning**) algorithm:
- **No explicit training step:** During `fit()`, k-NN simply stores the dataset in memory.
- **Computation happens during prediction:** When `predict()` is called, k-NN calculates distances between the new test point and all stored training samples, finds the $k$ closest neighbors, and aggregates their target values.

---

## How k-NN Works

### 1. Classification (`KNeighborsClassifier`)
To classify a new sample $\mathbf{x}_{\text{new}}$:
1. Compute the distance between $\mathbf{x}_{\text{new}}$ and every training sample.
2. Select the $k$ nearest training samples (neighbors).
3. Perform a **majority vote** among the target labels of those $k$ neighbors. The most common label is assigned as the prediction.

### 2. Regression (`KNeighborsRegressor`)
To predict a continuous value for a new sample $\mathbf{x}_{\text{new}}$:
1. Compute distances to all training points.
2. Select the $k$ nearest neighbors.
3. Compute the **mean** (or weighted mean) of the target values of those $k$ neighbors.

---

## Distance Metrics

The definition of "nearness" depends on the distance metric used.

### 1. Euclidean Distance ($L_2$ norm) — *Default*
The straight-line distance between two points $\mathbf{u}$ and $\mathbf{v}$ in $n$-dimensional space:

$$d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}$$

### 2. Manhattan Distance ($L_1$ norm)
The sum of absolute differences across all dimensions (grid-like distance):

$$d(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^{n} |u_i - v_i|$$

### 3. Minkowski Distance ($L_p$ norm)
A generalized distance metric parameterized by $p$:

$$d(\mathbf{u}, \mathbf{v}) = \left( \sum_{i=1}^{n} |u_i - v_i|^p \right)^{1/p}$$

- When $p = 1$, Minkowski distance reduces to **Manhattan distance**.
- When $p = 2$, Minkowski distance reduces to **Euclidean distance**.

---

## Choosing Hyperparameter $k$ (`n_neighbors`)

The choice of $k$ controls the **Bias-Variance Trade-off**:

```
      Small k (e.g., k=1)                  Optimal k (e.g., k=5)                Large k (e.g., k=50)
  High Variance / Overfitting                Balanced Model                    High Bias / Underfitting
+-----------------------------+     +-----------------------------+     +-----------------------------+
| Complex, noisy boundary     |     | Smooth, realistic boundary  |     | Over-simplified boundary    |
| Memorizes noise & outliers  |     | Captures underlying pattern |     | Ignores local structure     |
+-----------------------------+     +-----------------------------+     +-----------------------------+
```

- **Small $k$ (e.g., $k=1$):**
  - Decision boundary is highly sensitive to noise and outliers.
  - Low bias, high variance $\implies$ **Overfitting**.
- **Large $k$ (e.g., $k = N$):**
  - Decision boundary becomes overly smooth and predicts the majority class everywhere.
  - High bias, low variance $\implies$ **Underfitting**.
- **Rule of Thumb:** Start with $k = \sqrt{N}$ (preferably an **odd number** for binary classification to break voting ties).

---

## Distance Weighting (`weights`)

In standard k-NN (`weights='uniform'`), all $k$ nearest neighbors have equal influence on the prediction regardless of distance.

With `weights='distance'`, closer neighbors have a greater impact than farther ones:

$$w_i = \frac{1}{d(\mathbf{x}_{\text{new}}, \mathbf{x}_i)}$$

This is particularly useful when $k$ is larger or when density varies across feature space.

---

## Why Feature Scaling is MANDATORY for k-NN 🚨

Because k-NN relies entirely on distance calculations, **features with larger numerical scales will completely dominate distance metrics**.

### Example Scenario:
Imagine predicting customer churn using two features:
- **Age**: range $18 - 80$ (diff $\approx 60$)
- **Annual Income**: range $\$10,000 - \$200,000$ (diff $\approx 190,000$)

Without feature scaling, a difference of $\$1,000$ in income will completely dwarf a 30-year difference in age!

> **CRITICAL RULE:** Always scale features using `StandardScaler` or `MinMaxScaler` **before** fitting a k-NN model. Fit the scaler ONLY on training data!

---

## The Curse of Dimensionality

As the number of features (dimensions) $D$ increases:
1. The volume of feature space grows exponentially ($V \propto r^D$).
2. Data points become extremely sparse.
3. Distances between points become virtually uniform (the distance to the nearest neighbor approaches the distance to the farthest neighbor).
4. k-NN performance degrades significantly in high dimensions ($D > 20$).

**Mitigation:** Perform dimensionality reduction (e.g., PCA) or feature selection before applying k-NN on high-dimensional datasets.

---

## Scikit-Learn k-NN API

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Classification Pipeline
clf = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski', p=2)
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Regression Pipeline
reg = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor(n_neighbors=5, weights='distance')
)

reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)
```

---

## 🎤 Top Interview Questions

1. **Why is k-NN referred to as a "Lazy Learner"?**
   - *Answer*: Because it performs no actual training computation during `.fit()`. It simply stores the dataset in memory and defers all computations (distance calculations and neighbor sorting) until `.predict()` time.

2. **Why is feature scaling essential before using k-NN?**
   - *Answer*: k-NN calculates Euclidean/Manhattan distances between data points. Features with large numerical ranges will dominate the distance metric, rendering smaller-scale features irrelevant. Scaling ensures every feature contributes equally.

3. **What happens to the decision boundary of k-NN when $k=1$ vs when $k=N$?**
   - *Answer*: When $k=1$, the decision boundary is overly complex, flexible, and sensitive to noise/outliers (overfitting, high variance). When $k=N$ (total samples), the decision boundary predicts the majority class for all points (underfitting, high bias).

4. **How do you resolve ties in k-NN binary classification?**
   - *Answer*: Choose an **odd value for $k$** (e.g., $k=3, 5, 7$) when performing binary classification, or use distance-weighted voting (`weights='distance'`).

5. **How does k-NN suffer from the Curse of Dimensionality, and how can you mitigate it?**
   - *Answer*: In high dimensions, data becomes sparse and distance metrics lose discriminative power as all points become nearly equidistant. Mitigation includes feature selection (`SelectKBest`, `RFE`) or feature extraction (`PCA`, `TruncatedSVD`) prior to running k-NN.
