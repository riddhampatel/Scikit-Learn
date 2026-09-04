# 📝 Day 14 — Support Vector Machines (SVM)

## What is a Support Vector Machine?
A **Support Vector Machine (SVM)** is a powerful, versatile supervised machine learning algorithm capable of performing:
1. **Linear and non-linear classification** (`SVC`, `LinearSVC`)
2. **Linear and non-linear regression** (`SVR`, `LinearSVR`)
3. **Outlier / Novelty detection** (`OneClassSVM`)

SVMs are particularly well suited for classification of complex, small-to-medium sized datasets, and high-dimensional feature spaces (e.g., bioinformatics, genomics, image classification).

---

## 1. The Core Intuition: Maximum Margin Classifier

In binary classification, many lines/hyperplanes can separate two linearly separable classes. However, not all decision boundaries generalize equally well.

```
       Class A (+)                       Class B (-)
           *                                 #
        *     *                           #     #
           *    [Support Vector]             #
                /                      /
               /   Optimal Hyperplane /
              /         |            /
             /          |           /
            /      <-- Margin -->  /
           /                      /
                                 [Support Vector]
```

### Key Concepts:
- **Decision Hyperplane:** An $(n-1)$-dimensional subspace that separates $n$-dimensional space into two classes:
  $$\mathbf{w}^T \mathbf{x} + b = 0$$
- **Margin:** The geometric distance between the decision hyperplane and the closest training data points from either class.
- **Maximum Margin Separator:** SVM seeks the unique hyperplane that **maximizes** this margin width ($2 / \|\mathbf{w}\|$). A wider margin provides higher confidence and better generalization to unseen test data.
- **Support Vectors:** The critical training instances that lie exactly on the boundary of the margin (or violate it). The decision boundary is **entirely determined** by these support vectors. Removing all other data points would not change the decision boundary at all!

---

## 2. Hard Margin vs Soft Margin (Slack Variables & $C$)

### Hard Margin Classification
- **Condition:** Strictly requires *every single* data point to be correctly classified and strictly outside the margin.
- **Issues:**
  1. Only works if data is strictly linearly separable.
  2. Extremely sensitive to outliers (a single outlier can severely distort the hyperplane or make separation impossible).

### Soft Margin Classification (Slack Variables $\xi_i$)
To handle noisy, non-separable real-world data, SVM introduces **slack variables** $\xi_i \ge 0$, allowing some points to violate the margin or even be misclassified.

The optimization objective balances margin maximization with margin violation penalty:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i \quad \text{subject to } y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \; \xi_i \ge 0$$

### The Regularization Parameter $C$:
The hyperparameter `C` controls the trade-off between maximizing margin width and minimizing misclassifications:

| Hyperparameter | Margin Width | Tolerance for Violations | Model Complexity | Risk |
| :--- | :--- | :--- | :--- | :--- |
| **Small `C`** (e.g., $0.01, 0.1$) | **Wider margin** | High tolerance (more violations allowed) | Simpler model (High Bias) | **Underfitting** |
| **Large `C`** (e.g., $10, 100$) | **Narrow margin** | Low tolerance (fewer violations allowed) | Complex model (High Variance) | **Overfitting** |

> [!TIP]
> Notice the inverse behavior: In Scikit-Learn, `C` is a penalty on errors. Smaller `C` $\implies$ stronger regularization (softer margin); larger `C` $\implies$ weaker regularization (harder margin).

---

## 3. The Kernel Trick: Non-Linear Classification

When data cannot be separated by a straight line/hyperplane in the original feature space $\mathbb{R}^d$, we can project the data into a higher-dimensional feature space $\mathbb{R}^D$ ($D > d$) where the classes *become* linearly separable.

```
1D Line (Non-separable)          -->    2D Parabola Space (Linearly Separable)
   [ -  -  +  +  +  -  - ]       -->           +   +   +
                                             -           -
                                           -               -
                                         --------------------- (Hyperplane)
```

### The Kernel Trick Breakthrough:
Explicitly transforming every data point $\phi(\mathbf{x})$ into high (or infinite) dimensions is computationally intractable. 

The **Kernel Trick** computes the dot product in the high-dimensional space directly using a **Kernel function** $K(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$ without ever calculating the transformed coordinates explicitly!

---

## 4. Major Kernel Functions in Scikit-Learn

### 1. Linear Kernel (`kernel='linear'`)
$$K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}$$
- **When to use:** Linearly separable data, or very high-dimensional feature spaces where $N_{\text{features}} \gg N_{\text{samples}}$ (e.g. text Bag-of-Words, genomics).
- Fast and interpretable (has direct `coef_` weights).

### 2. Polynomial Kernel (`kernel='poly'`)
$$K(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^T \mathbf{z} + r)^d$$
- **Parameters:** `degree` ($d$, default 3), `gamma` ($\gamma$), `coef0` ($r$, default 0).
- Models polynomial interactions between features up to degree $d$.

### 3. Radial Basis Function (RBF) / Gaussian Kernel (`kernel='rbf'`)
$$K(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma \|\mathbf{x} - \mathbf{z}\|^2\right)$$
- **Default and most versatile kernel in Scikit-Learn.**
- Implicitly maps data into an **infinite-dimensional** Hilbert space.
- Measures similarity based on Euclidean distance between sample pairs.

### 4. Sigmoid Kernel (`kernel='sigmoid'`)
$$K(\mathbf{x}, \mathbf{z}) = \tanh(\gamma \mathbf{x}^T \mathbf{z} + r)$$
- Mimics a two-layer artificial neural network (perceptron).

---

## 5. The Kernel Coefficient: `gamma` ($\gamma$)

In the RBF and Polynomial kernels, `gamma` defines the **radius of influence** of each support vector:

$$K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{z}\|^2}{2\sigma^2}\right) \quad \text{where } \gamma = \frac{1}{2\sigma^2}$$

```
Small gamma (Large sigma):                     Large gamma (Small sigma):
Broad bell curve, wide influence               Narrow spike, local influence
Smooth, gentle decision boundary               Wiggly, tight decision boundary around points
Risk: UNDERFITTING (High Bias)                 Risk: OVERFITTING (High Variance)
```

### Options in Scikit-Learn:
- `gamma='scale'` (default): $\gamma = \frac{1}{n_{\text{features}} \cdot \text{Var}(X)}$
- `gamma='auto'`: $\gamma = \frac{1}{n_{\text{features}}}$
- Explicit float (e.g., `gamma=0.01`, `gamma=1.0`)

---

## 6. Support Vector Regression (`SVR`)

Instead of trying to fit the largest margin between classes without violations, **Support Vector Regression (SVR)** flips the objective:
> SVR tries to fit as many instances as possible **inside** a margin boundary ("$\epsilon$-tube"), while limiting margin violations (points falling outside the tube).

```
          y ^
            |          * (Violation: Slack xi_i*)
            |       ----------------------- + epsilon
            |      /   *       *    *     /
            |     /  *    *   f(x)   *   /   Optimal Regression Line f(x)
            |    /     *   *       *    /
            |   ----------------------- - epsilon
            |        * (Violation: Slack xi_i)
            +-----------------------------------> X
```

### Key Parameters:
- `epsilon` ($\epsilon$): Defines the width of the no-penalty tolerance tube. Errors smaller than $\epsilon$ are completely ignored in the loss function ($\epsilon$-insensitive loss).
- `C`: Penalty for samples that fall outside the $\epsilon$-tube.
- `kernel`: `linear`, `rbf`, `poly`.

---

## 7. ⚠️ Feature Scaling is MANDATORY for SVM

Because SVM calculates geometric Euclidean distances and dot products between feature vectors:
1. Features with large scales (e.g., Salary in \$50,000s) will completely dominate distance metrics over features with small scales (e.g., Age in 20-60s).
2. The margin will be distorted along large-scale axes, degrading classification performance.
3. Optimization algorithms (`libsvm` solver) take significantly longer to converge on unscaled data.

> [!IMPORTANT]
> **Always apply `StandardScaler` (or `MinMaxScaler`) before fitting any SVM estimator!**

---

## 8. Summary of Scikit-Learn SVM API

| Estimator | Typical Use | Key Hyperparameters | Key Attributes |
| :--- | :--- | :--- | :--- |
| `SVC` | General Classification | `C`, `kernel`, `degree`, `gamma`, `probability` | `support_`, `support_vectors_`, `n_support_`, `dual_coef_` |
| `LinearSVC` | Fast Linear Classification | `C`, `loss`, `penalty`, `dual` | `coef_`, `intercept_` |
| `SVR` | Non-Linear Regression | `C`, `epsilon`, `kernel`, `gamma` | `support_`, `support_vectors_`, `dual_coef_` |
| `LinearSVR` | Fast Linear Regression | `C`, `epsilon`, `loss` | `coef_`, `intercept_` |

---

## 9. Strengths vs Limitations

### ✅ Advantages:
1. **Effective in High Dimensions:** Exceptional performance when number of features $D$ exceeds number of samples $N$.
2. **Memory Efficient:** Only a subset of training points (support vectors) are stored in memory for the decision function.
3. **Versatile:** Wide selection of kernels (Linear, Poly, RBF, custom) to model complex non-linear boundaries.
4. **Robust to Outliers:** Points far away from the margin do not influence the position of the decision boundary.

### ❌ Limitations:
1. **Computational Complexity:** Training time scales between $O(N^2 \cdot D)$ and $O(N^3 \cdot D)$. Unsuitable for very large datasets ($N > 100,000$ samples) — use `LinearSVC` or `SGDClassifier` instead.
2. **Feature Scaling Sensitive:** Requires careful scaling (`StandardScaler`).
3. **Black Box & Probability Computation:** Kernels make interpretation difficult. Calibrated probabilities require expensive 5-fold cross-validation (`probability=True` uses Platt scaling).
