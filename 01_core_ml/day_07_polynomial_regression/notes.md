# 📝 Day 07 — Polynomial Regression & Overfitting

## What is it?
Polynomial Regression extends Linear Regression by generating higher-order feature terms ($x^2, x^3, \dots, x^d$) and interaction terms ($x_1 x_2$), allowing the model to capture **nonlinear** relationships while still using the linear OLS solver under the hood.

---

## Core Math & Concepts

### 1. Feature Expansion
For a single feature $x$ and polynomial degree $d$:
$$x \longrightarrow [1,\; x,\; x^2,\; x^3,\; \dots,\; x^d]$$

For two features $x_1, x_2$ at degree 2:
$$[x_1, x_2] \longrightarrow [1,\; x_1,\; x_2,\; x_1^2,\; x_1 x_2,\; x_2^2]$$

The model is still **linear in its weights** $w$, just nonlinear in the original features.

### 2. Bias-Variance Tradeoff
| Concept | Low Degree (e.g. 1) | High Degree (e.g. 15) |
| :--- | :--- | :--- |
| **Bias** | High — underfits the data | Low — fits training data tightly |
| **Variance** | Low — stable across datasets | High — wildly changes across datasets |
| **Train Error** | High | Very low (near zero) |
| **Test Error** | High | Very high (explodes) |

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

- **Underfitting** (high bias): Model is too simple to capture the true pattern.
- **Overfitting** (high variance): Model memorizes training noise instead of learning the true pattern.
- **Sweet Spot**: The degree $d$ where test error is minimized.

### 3. Detecting Overfitting
- **Train R² ≈ 1.0** but **Test R² << Train R²** (or even negative) → Overfitting.
- **Train RMSE ≈ 0** but **Test RMSE >> Train RMSE** → Overfitting.
- The gap between train and test metrics is the key diagnostic signal.

---

## Scikit-Learn `PolynomialFeatures` API

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Generate polynomial + interaction features up to degree d
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

# Or use a Pipeline (cleaner)
poly_model = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    LinearRegression()
)
poly_model.fit(X_train, y_train)
y_pred = poly_model.predict(X_test)
```

### Key Parameters
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `degree` | `2` | Maximum polynomial degree |
| `interaction_only` | `False` | If `True`, only interaction terms (no $x_i^2$) |
| `include_bias` | `True` | If `True`, includes a column of ones (intercept term). Set `False` when using `LinearRegression` since it adds its own intercept. |

---

## 🎤 Top Interview Questions

1. **Is Polynomial Regression a linear or nonlinear model?**
   - *Answer*: It is a **linear** model. It is nonlinear in the original features but linear in its parameters (weights). After feature expansion, it applies ordinary linear regression on the expanded feature matrix.

2. **How do you detect overfitting in Polynomial Regression?**
   - *Answer*: Compare train vs test metrics (R², RMSE). A large gap where train performance is excellent but test performance degrades indicates overfitting. Additionally, very large coefficient magnitudes are a symptom of overfitting.

3. **What is the Bias-Variance Tradeoff?**
   - *Answer*: Bias measures how far the model's average prediction is from the true value (underfitting). Variance measures how much predictions fluctuate across different training sets (overfitting). Total error is bias² + variance + irreducible noise. The goal is to find the model complexity that minimizes total error.

4. **Why does `include_bias=False` matter when using `LinearRegression`?**
   - *Answer*: `LinearRegression` already adds an intercept term via `fit_intercept=True` (default). If `PolynomialFeatures` also adds a bias column of ones, the intercept is duplicated, causing a redundant (collinear) column. Setting `include_bias=False` avoids this.
