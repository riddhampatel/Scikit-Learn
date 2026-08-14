# 📝 Day 08 — Ridge Regression (L2 Regularization)

## What is it?
**Ridge Regression** is a regularized variant of Linear Regression that adds an **L2 penalty** (squared magnitude of coefficients) to the ordinary least squares (OLS) loss function. It prevents **overfitting** and mitigates **multicollinearity** (highly correlated features) by shrinking coefficient magnitudes towards zero.

---

## Core Math & Concepts

### 1. Cost Function
Ordinary Least Squares (OLS) minimizes Mean Squared Error (MSE):
$$J_{\text{OLS}}(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (\mathbf{w}^T \mathbf{x}_i + b))^2$$

Ridge Regression adds an **L2 regularization penalty**:
$$J_{\text{Ridge}}(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (\mathbf{w}^T \mathbf{x}_i + b))^2 + \alpha \sum_{j=1}^{p} w_j^2$$

In matrix vector notation:
$$J_{\text{Ridge}}(\mathbf{w}) = \frac{1}{n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_2^2$$

> ⚠️ **Note:** The intercept $b$ (or $w_0$) is **never** penalized, because shrinking the intercept would force the line towards the origin regardless of the data's mean.

---

### 2. Closed-Form Analytical Solution
- **OLS Solution:**
  $$\hat{\mathbf{w}}_{\text{OLS}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
  *(Fails or becomes unstable when $\mathbf{X}^T \mathbf{X}$ is singular / collinear).*

- **Ridge Solution:**
  $$\hat{\mathbf{w}}_{\text{Ridge}} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$
  *Adding $\alpha \mathbf{I}$ guarantees that $(\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})$ is invertible even if $\mathbf{X}^T \mathbf{X}$ is singular.*

---

### 3. Hyperparameter $\alpha$ (Alpha)
The parameter $\alpha \ge 0$ controls the trade-off between fitting the data and keeping weights small:

| $\alpha$ Value | Behavior | Model Complexity | Risk |
| :--- | :--- | :--- | :--- |
| $\alpha = 0$ | Equivalent to Standard OLS Linear Regression | High | Overfitting |
| Small $\alpha$ (e.g. $0.01$) | Mild regularization; suppresses extreme weights | Optimal / Balanced | Good generalization |
| Large $\alpha$ (e.g. $1000$) | Heavy regularization; forces weights $w_j \to 0$ | Low | Underfitting |
| $\alpha \to \infty$ | All coefficients $w_j \to 0$; prediction becomes flat mean line | Minimal | Severe Underfitting |

---

### 4. Coefficient Shrinkage vs Feature Selection
- Ridge performs **coefficient shrinkage**, **NOT** feature selection.
- As $\alpha$ increases, coefficients asymptotically approach zero, but **never become exactly 0**.
- All features remain in the final model (unlike Lasso / L1 regularization).

---

### 5. Why Feature Scaling is Mandatory 🚨
The L2 penalty term $\alpha \sum w_j^2$ treats all coefficients equally. If features are on different scales (e.g., age in years $[0–100]$ vs income in dollars $[0–100,000]$):
- Features with larger numeric values naturally get smaller coefficients $w_j$.
- Features with smaller numeric values need larger coefficients $w_j$.
- Ridge would penalize smaller-scale features disproportionately!
- **Rule:** Always apply `StandardScaler` **before** fitting Ridge Regression!

---

## Scikit-Learn `Ridge` & `RidgeCV` API

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Standard Ridge Pipeline
ridge_pipe = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0, solver="auto", random_state=42)
)
ridge_pipe.fit(X_train, y_train)

# 2. Efficient Built-in Cross-Validation for Best Alpha
ridge_cv = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
)
ridge_cv.fit(X_train, y_train)
best_alpha = ridge_cv.named_steps["ridgecv"].alpha_
```

### Key Hyperparameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `alpha` | `1.0` | Regularization strength (must be non-negative float) |
| `solver` | `'auto'` | Algorithm for optimization (`'svd'`, `'cholesky'`, `'lsqr'`, `'sag'`, `'saga'`) |
| `fit_intercept` | `True` | Whether to calculate the intercept for this model |
| `max_iter` | `None` | Maximum number of iterations for conjugate gradient solver |
| `tol` | `1e-4` | Precision of the solution |

---

## OLS vs Ridge Comparison

| Aspect | OLS Linear Regression | Ridge Regression (L2) |
| :--- | :--- | :--- |
| **Loss Function** | $\text{MSE}$ | $\text{MSE} + \alpha \|\mathbf{w}\|_2^2$ |
| **Multicollinearity** | Unstable; exploding weights | Stable; handles collinearity well |
| **Overfitting** | Prone to overfitting on noisy/high-dim data | Controlled via $\alpha$ hyperparameter |
| **Coefficients** | Can be arbitrarily large | Continuously shrunken towards 0 |
| **Feature Selection** | None | None (keeps all features) |
| **Feature Scaling** | Optional (affects interpretation only) | **Mandatory** |

---

## 🎤 Top Interview Questions

1. **What is the mathematical difference between OLS and Ridge Regression?**
   - *Answer*: OLS minimizes pure Mean Squared Error. Ridge Regression minimizes MSE plus an L2 penalty proportional to the sum of squared coefficient weights ($\alpha \sum w_j^2$).

2. **Why does Ridge Regression help with multicollinearity?**
   - *Answer*: When features are collinear, $\mathbf{X}^T \mathbf{X}$ is near-singular, making $(\mathbf{X}^T \mathbf{X})^{-1}$ unstable with giant, unstable coefficients. Ridge adds $\alpha \mathbf{I}$ to $\mathbf{X}^T \mathbf{X}$, making the matrix strictly invertible and conditioning the problem so coefficient variances stay small.

3. **Does Ridge Regression perform feature selection?**
   - *Answer*: No. Ridge shrinks coefficient values continuously towards zero as $\alpha$ increases, but coefficients never reach absolute zero. All features are retained in the model. (Lasso / L1 is used for feature selection).

4. **Why is feature scaling essential before fitting Ridge Regression?**
   - *Answer*: The penalty term treats all weights $w_j^2$ equally. Unscaled features with small numerical ranges require larger weights to impact predictions, so Ridge penalizes them unfairly compared to features on large scales. Scaling ensures equal penalty weight across all features.

5. **How does alpha control the Bias-Variance Tradeoff in Ridge?**
   - *Answer*: Increasing $\alpha$ increases bias (simplifies model structure) while decreasing variance (stabilizes prediction fluctuation across training sets). Decreasing $\alpha$ towards 0 decreases bias but increases variance.
