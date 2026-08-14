# 📝 Day 09 — Lasso Regression & ElasticNet

## What is Lasso Regression?
**Lasso Regression** (Least Absolute Shrinkage and Selection Operator) is a regularized regression method that adds an **L1 penalty** (sum of absolute values of weights) to the ordinary least squares (OLS) loss function.

Unlike Ridge (L2 penalty) which shrinks weights towards zero without making them exactly zero, Lasso shrinks small/unimportant feature coefficients **strictly to zero**. This gives Lasso a unique property: **built-in automatic feature selection** and sparse model solutions.

---

## What is ElasticNet?
**ElasticNet** combines both **L1 (Lasso)** and **L2 (Ridge)** penalties in a single objective function.

It overcomes key limitations of Lasso:
1. When features are highly correlated, Lasso tends to randomly select only one feature from the group and ignore the rest. ElasticNet retains grouped correlated features (like Ridge).
2. When the number of features $p > n$ (more features than data samples), Lasso selects at most $n$ features. ElasticNet can select more than $n$ features.

---

## Core Math & Concepts

### 1. Lasso Cost Function (L1 Regularization)
$$J_{\text{Lasso}}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (\mathbf{w}^T \mathbf{x}_i + b))^2 + \alpha \sum_{j=1}^{p} |w_j|$$

In matrix-vector notation:
$$J_{\text{Lasso}}(\mathbf{w}) = \frac{1}{n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_1$$

Where:
- $\|\mathbf{w}\|_1 = \sum_{j=1}^p |w_j|$ is the **L1 norm**.
- $\alpha \ge 0$ controls regularization strength.

> ⚠️ **Note:** The derivative of $|w_j|$ is undefined at $w_j = 0$. Therefore, Lasso cannot be solved with a simple closed-form linear algebra equation like OLS or Ridge. Instead, it is solved iteratively using **Coordinate Descent**.

---

### 2. ElasticNet Cost Function (L1 + L2 Regularization)
$$J_{\text{ElasticNet}}(\mathbf{w}) = \frac{1}{n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \cdot \text{l1\_ratio} \cdot \|\mathbf{w}\|_1 + \frac{\alpha (1 - \text{l1\_ratio})}{2} \|\mathbf{w}\|_2^2$$

Where:
- $\alpha \ge 0$: Total regularization penalty multiplier.
- $\text{l1\_ratio} \in [0, 1]$: Mixing parameter between L1 and L2 penalties:
  - $\text{l1\_ratio} = 1 \implies \text{Pure Lasso (L1)}$
  - $\text{l1\_ratio} = 0 \implies \text{Pure Ridge (L2)}$
  - $0 < \text{l1\_ratio} < 1 \implies \text{ElasticNet (L1 + L2 combined)}$

---

### 3. Why L1 Regularization Causes Sparsity (Feature Selection)

#### Geometric Intuition
Consider minimizing loss $MSE$ subject to a constraint on weights:
- **Ridge (L2 Constraint):** Represented by a smooth sphere/circle $\|\mathbf{w}\|_2^2 \le C$. The loss contours intersect the circle at arbitrary non-zero points along the boundary.
- **Lasso (L1 Constraint):** Represented by a diamond/polytope with sharp corners along the coordinate axes $\|\mathbf{w}\|_1 \le C$. The elliptical loss contours typically intersect the diamond directly at one of its sharp corners (where one or more $w_j = 0$).

```
       Ridge (L2 Circle)                  Lasso (L1 Diamond)
             w2                                 w2
             |                                  |
          .  |  .                             /\
        .    |    .                          /  \
       ------|------ w1                  ---|----+--- w1
        .    |    .                          \  /
          .  |  .                             \/
             |                                  |
(Intersects anywhere on curve)      (Intersects at corners where w1=0 or w2=0)
```

---

## Model Selection Guide: When to Use Which?

| Scenario | Recommended Model | Why? |
| :--- | :--- | :--- |
| Few features, all believed to be relevant | **Ridge (L2)** | Preserves all features, controls multicollinearity |
| High-dimensional data with many noisy/irrelevant features | **Lasso (L1)** | Zeroes out irrelevant features, builds simple/interpretable model |
| High-dimensional data with correlated feature groups | **ElasticNet** | Performs feature selection while keeping groups of correlated variables |
| $p > n$ (features outnumber samples) | **ElasticNet / Ridge** | Lasso breaks down / caps selection at $n$ features |

---

## Scikit-Learn API & Hyperparameters

```python
from sklearn.linear_model import Lasso, ElasticNet, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Lasso Model
lasso_pipe = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.1, max_iter=10000, random_state=42)
)
lasso_pipe.fit(X_train, y_train)

# 2. ElasticNet Model
elastic_pipe = make_pipeline(
    StandardScaler(),
    ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42)
)
elastic_pipe.fit(X_train, y_train)

# 3. LassoCV (Automatic Alpha Tuning)
lasso_cv = make_pipeline(
    StandardScaler(),
    LassoCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5, random_state=42)
)
lasso_cv.fit(X_train, y_train)

# 4. ElasticNetCV (Automatic Alpha & L1_ratio Tuning)
elastic_cv = make_pipeline(
    StandardScaler(),
    ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99], cv=5, random_state=42)
)
elastic_cv.fit(X_train, y_train)
```

### Parameter Reference

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `alpha` | `1.0` | Constant that multiplies the penalty terms |
| `l1_ratio` | `0.5` | ElasticNet penalty mixing parameter ($0 \le \text{l1\_ratio} \le 1$) |
| `max_iter` | `1000` | Maximum iterations for coordinate descent convergence |
| `tol` | `1e-4` | Convergence tolerance |
| `selection` | `'cyclic'` | If `'random'`, updates coefficients in random order each iteration (speeds up convergence) |

---

## Comprehensive Model Comparison Table

| Property | OLS Linear | Ridge (L2) | Lasso (L1) | ElasticNet |
| :--- | :--- | :--- | :--- | :--- |
| **Penalty Term** | None | $\alpha \|\mathbf{w}\|_2^2$ | $\alpha \|\mathbf{w}\|_1$ | $\alpha \cdot \text{l1\_ratio} \|\mathbf{w}\|_1 + \frac{\alpha(1-\text{l1\_ratio})}{2} \|\mathbf{w}\|_2^2$ |
| **Solution Method** | Closed-form | Closed-form | Coordinate Descent | Coordinate Descent |
| **Feature Selection** | No | No | **Yes** (exact zeros) | **Yes** (exact zeros) |
| **Sparsity** | No | No | High | Controlled by `l1_ratio` |
| **Correlated Features** | Unstable | Retains all | Picks 1 arbitrarily | Retains feature group |
| **Scaling Required?** | Optional | **Mandatory** | **Mandatory** | **Mandatory** |

---

## 🎤 Top Interview Questions

1. **Why does L1 regularization cause sparse solutions (exact zero coefficients), whereas L2 does not?**
   - *Answer*: Mathematically, the L1 penalty derivative is constant ($\pm \alpha$), whereas the L2 penalty derivative is proportional to weight magnitude ($2\alpha w_j$). Geometrically, the L1 constraint region is a diamond with sharp corners on the coordinate axes. Optimization loss contours hit these sharp corners directly, forcing weights to zero.

2. **When should you choose ElasticNet over pure Lasso?**
   - *Answer*: Use ElasticNet when features are strongly correlated or when $p > n$ (features outnumber samples). Lasso will pick one arbitrary feature out of a set of correlated variables and drop the rest, while ElasticNet groups correlated features together and avoids capping selection at $n$.

3. **What happens to a Lasso model as $\alpha \to \infty$?**
   - *Answer*: As $\alpha$ increases to infinity, the L1 penalty dominates completely. All feature coefficients $w_j$ are driven to exact zero, leaving only the intercept $b$ (which predicts the sample mean of $y$).

4. **Can Lasso be solved analytically using matrix inversion? Why or why not?**
   - *Answer*: No. The absolute value function $|w_j|$ in the L1 penalty is non-differentiable at $w_j = 0$. Therefore, there is no closed-form analytical equation $(\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$. Lasso relies on numerical iterative solvers such as **Coordinate Descent**.

5. **How do `alpha` and `l1_ratio` interact in Scikit-Learn's `ElasticNet`?**
   - *Answer*: `alpha` controls the overall magnitude of regularization applied to the model. `l1_ratio` determines how that penalty budget is divided between L1 (Lasso) and L2 (Ridge). Setting `l1_ratio=1.0` yields pure Lasso, while `l1_ratio=0.0` yields pure Ridge.
