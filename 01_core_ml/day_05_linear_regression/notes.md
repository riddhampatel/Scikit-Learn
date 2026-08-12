# 📝 Day 05 — Linear Regression Baseline

## What is it?
Linear Regression is the fundamental baseline algorithm for continuous regression tasks. It models the linear relationship between a continuous dependent target variable ($y$) and one or more independent feature variables ($X$).

---

## Core Math & Concepts

### 1. Mathematical Equation
For $n$ features, the linear prediction $\hat{y}$ is calculated as:
$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = Xw + b$$

- **$w$ (`model.coef_`)**: Feature weights / coefficients (slopes). Indicates how much $y$ changes per unit change in $x_i$.
- **$b$ (`model.intercept_`)**: Bias / intercept. Expected value of $y$ when all features $x_i = 0$.

### 2. Ordinary Least Squares (OLS) Optimization
Scikit-Learn's `LinearRegression` finds weights $w$ by minimizing the **Sum of Squared Residuals (SSR)**:
$$\text{SSR} = \sum_{i=1}^{M} (y_i - \hat{y}_i)^2$$

It computes the closed-form normal equation analytical solution:
$$w = (X^T X)^{-1} X^T y$$

---

## Scikit-Learn `LinearRegression` API

```python
from sklearn.linear_model import LinearRegression

# 1. Instantiate
model = LinearRegression(fit_intercept=True)

# 2. Train / Fit
model.fit(X_train, y_train)

# 3. Inspect learned parameters
print("Coefficients (w):", model.coef_)
print("Intercept (b):", model.intercept_)

# 4. Predict & Residuals
y_pred = model.predict(X_test)
residuals = y_test - y_pred
```

---

## 🎤 Top Interview Questions

1. **What are Residuals in Linear Regression?**
   - *Answer*: A residual is the vertical difference between the actual target value $y_i$ and the predicted value $\hat{y}_i$ ($\text{residual} = y_i - \hat{y}_i$).

2. **Does Linear Regression require feature scaling (e.g. `StandardScaler`)?**
   - *Answer*: Mathematically, OLS closed-form solution yields the exact same predictions with or without scaling. However, scaling makes coefficient sizes directly comparable to assess feature importance.

3. **What is the difference between `coef_` shape for 1D vs multi-feature inputs?**
   - *Answer*: For $N$ features, `coef_` is a 1D array of shape `(N,)` containing one weight per feature. If $y$ is multi-output, `coef_` is a 2D array of shape `(n_targets, N)`.

4. **What are the key assumptions of OLS Linear Regression?**
   - *Answer*: Linearity, Independence of errors, Homoscedasticity (constant residual variance), Normality of residuals, and No Multicollinearity.
