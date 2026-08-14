# 📝 Day 10 — Logistic Regression (Binary Classification)

## What is Logistic Regression?
Despite having "Regression" in its name, **Logistic Regression** is a fundamental **supervised classification algorithm** used for predicting categorical outcomes. 

In **Binary Classification**, the goal is to map input features $\mathbf{x}$ to a target label $y \in \{0, 1\}$ (e.g., Benign vs. Malignant tumor, Spam vs. Not Spam, Default vs. No Default). Instead of fitting a straight line to continuous output like Linear Regression, Logistic Regression models the **probability** $p = P(y=1|\mathbf{x})$ that a given input belongs to the positive class.

---

## Core Math & Concepts

### 1. The Sigmoid (Logistic) Function
To convert any linear combination of features $z = \mathbf{w}^T \mathbf{x} + b$ (which ranges from $-\infty$ to $+\infty$) into a probability bounded strictly between $0$ and $1$, Logistic Regression applies the **Sigmoid function** $g(z)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- When $z \to +\infty$, $\sigma(z) \to 1$
- When $z \to -\infty$, $\sigma(z) \to 0$
- When $z = 0$, $\sigma(z) = 0.5$

```
          1.0 |                   /---
              |                 /
  Probability |               /  
        P(y=1)|             /    
          0.5 |------------*------------  (Decision Boundary at z=0)
              |          /
              |        /
          0.0 |___---/
              +------------------------
                 -inf     0     +inf
                            z
```

---

### 2. Log-Odds (Logit Function)
The relationship between input features and class probability can be written in terms of **Odds** and **Log-Odds (Logit)**:

$$\text{Odds} = \frac{p}{1 - p}$$

$$\text{Logit}(p) = \ln\left(\frac{p}{1 - p}\right) = \mathbf{w}^T \mathbf{x} + b$$

- A unit increase in feature $x_j$ multiplies the odds of the positive class by $e^{w_j}$.

---

### 3. Cost Function: Log Loss (Binary Cross-Entropy)
Linear Regression's Mean Squared Error (MSE) is non-convex when combined with the Sigmoid function, causing local minima during optimization. 

Instead, Logistic Regression uses **Log Loss (Binary Cross-Entropy)**:

$$J(\mathbf{w}, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

Where:
- $y_i \in \{0, 1\}$ is the true class label.
- $\hat{p}_i = \sigma(\mathbf{w}^T \mathbf{x}_i + b)$ is the predicted probability.

**Key Intuition:**
- If $y_i = 1$ and $\hat{p}_i \to 1$, loss approaches $0$. If $\hat{p}_i \to 0$, loss penalizes exponentially towards $+\infty$.
- If $y_i = 0$ and $\hat{p}_i \to 0$, loss approaches $0$. If $\hat{p}_i \to 1$, loss penalizes exponentially towards $+\infty$.

---

### 4. Decision Boundaries & `predict()` vs `predict_proba()`

- **`predict_proba(X)`**: Returns an array of predicted probabilities $[P(y=0), P(y=1)]$ for each sample.
- **`predict(X)`**: Applies a decision threshold (default = $0.5$) to classify samples into discrete labels:

$$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|\mathbf{x}) \ge 0.5 \\ 0 & \text{if } P(y=1|\mathbf{x}) < 0.5 \end{cases}$$

The **Decision Boundary** occurs where $z = \mathbf{w}^T \mathbf{x} + b = 0$, dividing the feature space into class 0 and class 1 regions.

---

## Scikit-Learn `LogisticRegression` API

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Building a pipeline with feature scaling and Logistic Regression
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
```

### Key Hyperparameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `C` | `1.0` | **Inverse of regularization strength** ($C = \frac{1}{\lambda}$). Smaller values $\implies$ stronger regularization (smaller weights). Larger values $\implies$ weaker regularization (fits data closely). |
| `penalty` | `'l2'` | Regularization type (`'l2'`, `'l1'`, `'elasticnet'`, or `None`). |
| `solver` | `'lbfgs'` | Optimization algorithm (`'lbfgs'`, `'liblinear'`, `'saga'`, `'newton-cg'`, `'sag'`). Note: `'liblinear'` supports L1 penalty for small datasets; `'saga'` supports L1/ElasticNet for large datasets. |
| `max_iter` | `100` | Maximum iterations for optimization algorithm to converge. Increase to $1000$ if convergence warnings occur. |
| `class_weight` | `None` | Set to `'balanced'` to handle imbalanced target distributions automatically. |
| `random_state` | `None` | Controls random seed for solver initialization. |

---

## Why Feature Scaling is Essential
- Logistic Regression solvers (like `'lbfgs'`, `'saga'`) rely on gradient-based optimization algorithms.
- Unscaled features lead to elongated cost function contours, slowing down gradient convergence.
- Regularization penalties ($L1$ or $L2$) apply uniformly across weights, so features with larger ranges would be penalized disproportionately.
- **Always apply `StandardScaler` before fitting `LogisticRegression`!**

---

## Evaluation Metrics for Classification

| Metric | Scikit-Learn Function | Description |
| :--- | :--- | :--- |
| **Accuracy** | `accuracy_score(y_true, y_pred)` | Ratio of correct predictions overall: $\frac{TP + TN}{TP + TN + FP + FN}$. |
| **Confusion Matrix** | `confusion_matrix(y_true, y_pred)` | Table showing True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN). |
| **Classification Report** | `classification_report(y_true, y_pred)` | Summary table with Precision, Recall, F1-Score, and Support for each class. |
| **ROC-AUC Score** | `roc_auc_score(y_true, y_proba[:, 1])` | Area under the Receiver Operating Characteristic curve (TPR vs FPR across all thresholds). |

---

## 🎤 Top Interview Questions

1. **Why is Logistic Regression called a regression model when it is used for classification?**
   - *Answer*: It is called regression because it estimates the parameters of a continuous linear combination of input features $z = \mathbf{w}^T \mathbf{x} + b$ using the log-odds transform (logit function), predicting a continuous probability value before thresholding into discrete classes.

2. **How does hyperparameter `C` in `LogisticRegression` differ from `alpha` in `Ridge`/`Lasso`?**
   - *Answer*: `C` is the **inverse** of regularization strength ($C = \frac{1}{\alpha}$). Thus, a **smaller** $C$ increases regularization (shrinking weights towards 0 to reduce overfitting), while a **larger** $C$ reduces regularization.

3. **What is the difference between `predict()` and `predict_proba()`?**
   - *Answer*: `predict()` returns discrete class labels (e.g., $0$ or $1$) based on a default threshold of $0.5$. `predict_proba()` returns a 2D array of continuous class probabilities $[P(y=0), P(y=1)]$, allowing custom threshold selection for imbalanced datasets or risk-sensitive applications.

4. **Why is Log Loss (Binary Cross-Entropy) preferred over Mean Squared Error (MSE) for Logistic Regression?**
   - *Answer*: When the Sigmoid function is plugged into MSE, the cost surface becomes non-convex with multiple local minima, making gradient descent unreliable. Log Loss forms a smooth, convex cost function where incorrect confident predictions are penalized exponentially, guaranteeing convergence to the global minimum.

5. **When should you set `class_weight='balanced'` in `LogisticRegression`?**
   - *Answer*: You should set `class_weight='balanced'` when working with imbalanced datasets (e.g. 95% benign, 5% malignant tumors). It automatically adjusts weights inversely proportional to class frequencies, preventing the model from biasing predictions toward the majority class.
