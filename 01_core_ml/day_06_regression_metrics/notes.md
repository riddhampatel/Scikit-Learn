# 📝 Day 06 — Regression Evaluation Metrics (MAE, MSE, RMSE, R² & Adjusted R²)

## What is it?
Evaluation metrics quantify how well a regression model's predictions ($\hat{y}$) match actual target values ($y$). Choosing the right metric depends on whether your domain penalizes large errors heavily or requires robust metrics resilient to outliers.

---

## Core Formulas & Intuition

### 1. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- **Units**: Same units as target $y$.
- **Intuition**: Average linear error magnitude. Treats all errors equally (robust to outliers).

### 2. Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
- **Units**: Squared target units ($y^2$).
- **Intuition**: Squares residual errors, heavily penalizing large outliers.

### 3. Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
- **Units**: Same units as target $y$.
- **Intuition**: Brings squared error penalty back to target scale. Standard metric for most ML competitions.

### 4. Coefficient of Determination ($R^2$ Score)
$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
- **Range**: $(-\infty, 1.0]$
  - $1.0$: Perfect predictions.
  - $0.0$: Predicts the simple target mean $\bar{y}$ for every sample.
  - $< 0.0$: Model performs *worse* than predicting the mean!

### 5. Adjusted $R^2$ Score
$$\text{Adjusted } R^2 = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - p - 1} \right]$$
- Where $n = \text{number of samples}$, $p = \text{number of features}$.
- **Why?**: Standard $R^2$ *always* increases when adding more features (even useless noise). Adjusted $R^2$ penalizes adding irrelevant features.

---

## Scikit-Learn `sklearn.metrics` API

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred) # sklearn >= 1.4
r2 = r2_score(y_true, y_pred)
```

---

## 🎤 Top Interview Questions

1. **When should you prefer MAE over RMSE?**
   - *Answer*: Prefer MAE when your dataset contains noisy outliers that you do not want to heavily influence the model evaluation score. Prefer RMSE when large prediction errors are catastrophic (e.g. medical dosage, financial risk).

2. **Why can $R^2$ score be negative?**
   - *Answer*: $R^2$ compares model residual variance against baseline target variance around the mean ($\bar{y}$). If the model predictions are completely uncalibrated and yield higher sum of squared errors than simply predicting $\bar{y}$, $R^2$ becomes negative.

3. **What is the difference between $R^2$ and Adjusted $R^2$?**
   - *Answer*: $R^2$ measures proportion of variance explained by features. Adjusted $R^2$ modifies $R^2$ by adding a penalty for every extra feature ($p$), ensuring model score drops if a feature adds no predictive value.
