# 📝 Day 15 — Decision Trees

## What is a Decision Tree?
A **Decision Tree** is a non-parametric supervised machine learning algorithm used for both **classification** (`DecisionTreeClassifier`) and **regression** (`DecisionTreeRegressor`).

Unlike linear models (which construct a continuous global equation $y = \mathbf{w}^T \mathbf{x} + b$) or instance-based models like k-NN (which store all points and calculate distances), a Decision Tree builds an **intuitive, hierarchical rule-based model**. It partitions the feature space into axis-aligned rectangular regions through a sequence of orthogonal binary splits (`if-else` questions).

```
                      [ Root Node: Feature X1 <= 2.45? ]
                                   /      \
                           Yes   /          \   No
                               /              \
         [ Leaf: Class 0 ]              [ Node: Feature X2 <= 1.75? ]
                                                /        \
                                        Yes   /            \   No
                                            /                \
                             [ Leaf: Class 1 ]          [ Leaf: Class 2 ]
```

Decision trees are the foundational building blocks for modern ensemble methods like **Random Forests**, **Extra Trees**, and **Gradient Boosted Trees (XGBoost, LightGBM, CatBoost)**.

---

## 1. Anatomy of a Decision Tree

| Component | Definition | Role |
| :--- | :--- | :--- |
| **Root Node** | The top-level starting node containing all training instances. | Receives full dataset; picks first optimal split feature and threshold. |
| **Internal / Decision Node** | Any intermediate non-terminal node that tests an attribute condition. | Evaluates a single feature condition ($x_j \le t_m$) and branches left or right. |
| **Branch / Edge** | The path connecting parent nodes to child nodes. | Represents the outcome of the split condition (e.g. "True" or "False"). |
| **Leaf / Terminal Node** | A bottom-level node with no outgoing edges. | Holds the final class prediction (mode) or regression target (mean/median). |
| **Tree Depth** | The longest path from the root node to any leaf node. | Governs model complexity: depth 1 = Decision Stump; infinite depth = pure memorization. |

---

## 2. How Decision Trees Learn: The CART Algorithm

Scikit-Learn implements an optimized version of the **CART (Classification and Regression Trees)** algorithm.

### Greedy Recursive Binary Splitting
Building a globally optimal tree is an NP-complete problem. CART uses a **greedy top-down heuristic**:
1. At the current node $m$, search across **all features** $j \in \{1, \dots, D\}$ and **all candidate thresholds** $t \in \mathbb{R}$.
2. Select the pair $(j, t)$ that produces the purest left child node $D_{\text{left}}(j, t)$ and right child node $D_{\text{right}}(j, t)$.
3. Partition the dataset into $D_{\text{left}}$ and $D_{\text{right}}$.
4. Recursively repeat the procedure on each child node.
5. Stop when a stopping criterion is reached (e.g., maximum depth, minimum sample limit, pure node).

### Split Optimization Objective
At node $m$, CART minimizes the weighted impurity of the two children:

$$G(D, j, t) = \frac{N_{\text{left}}}{N_m} I(D_{\text{left}}) + \frac{N_{\text{right}}}{N_m} I(D_{\text{right}})$$

where:
- $N_m$ is the number of samples in node $m$.
- $N_{\text{left}}, N_{\text{right}}$ are the sample counts in the left and right child subsets.
- $I(\cdot)$ is the impurity measure (Gini, Entropy, or MSE).

---

## 3. Classification Split Criteria

Scikit-Learn's `DecisionTreeClassifier` supports multiple impurity functions via the `criterion` parameter:

### 1. Gini Impurity (`criterion='gini'`, default)
Measures the probability that a randomly chosen element from the set would be incorrectly labeled if randomly labeled according to the label distribution in the subset:

$$I_{\text{Gini}}(m) = 1 - \sum_{k=1}^K p_{mk}^2$$

where $p_{mk}$ is the proportion of class $k$ instances in node $m$:
- **Pure Node ($I_{\text{Gini}} = 0$):** All instances belong to a single class ($p_{m1} = 1 \implies 1 - 1^2 = 0$).
- **Maximum Impurity:** Evenly balanced classes across $K$ classes ($I_{\text{Gini}} = 1 - \frac{1}{K}$). For binary classification, max Gini is $1 - (0.5^2 + 0.5^2) = 0.5$.

### 2. Information Gain & Entropy (`criterion='entropy'`)
Originating from Shannon's Information Theory, **Entropy** measures the average amount of surprise or disorder in the data distribution:

$$I_{\text{Entropy}}(m) = - \sum_{k=1}^K p_{mk} \log_2(p_{mk}) \quad (\text{with } 0 \log_2(0) \equiv 0)$$

The reduction in entropy achieved by splitting is called **Information Gain (IG)**:

$$\text{Information Gain} = I_{\text{Entropy}}(D) - \left[ \frac{N_{\text{left}}}{N_m} I_{\text{Entropy}}(D_{\text{left}}) + \frac{N_{\text{right}}}{N_m} I_{\text{Entropy}}(D_{\text{right}}) \right]$$

### Gini vs. Entropy Comparison

| Characteristic | Gini Impurity | Entropy / Information Gain |
| :--- | :--- | :--- |
| **Formula** | $1 - \sum p_k^2$ | $-\sum p_k \log_2(p_k)$ |
| **Computation Speed** | **Faster** (simple squares, no logarithms) | **Slower** (requires log calculations) |
| **Behavior** | Tends to isolate the most frequent class into its own branch. | Tends to produce slightly more balanced tree partitions. |
| **Practical Impact** | Differences in test performance are rarely significant (<2% in most benchmarks). |

### 3. Log Loss (`criterion='log_loss'`)
Calculates the Shannon entropy using the natural logarithm $\ln$ instead of $\log_2$.

---

## 4. Regression Split Criteria

In `DecisionTreeRegressor`, the target $y$ is continuous. Instead of measuring class impurity, CART measures **variance reduction**:

### 1. Mean Squared Error (`criterion='squared_error'`, default)
Measures the variance of the target values within node $m$:

$$I_{\text{MSE}}(m) = \frac{1}{N_m} \sum_{i \in m} (y_i - \bar{y}_m)^2 \quad \text{where } \bar{y}_m = \frac{1}{N_m} \sum_{i \in m} y_i$$

- The predicted value for any leaf node is the **mean** $\bar{y}_m$ of training samples falling into that leaf.
- Minimizing child weighted MSE maximizes variance reduction across splits.

### 2. Mean Absolute Error (`criterion='absolute_error'`)
Measures the $L_1$ deviation from the median:

$$I_{\text{MAE}}(m) = \frac{1}{N_m} \sum_{i \in m} |y_i - \text{median}(y_m)|$$

- The predicted leaf value is the **median**.
- Much more robust to extreme target outliers than squared error, but significantly slower to compute.

---

## 5. Overfitting & Regularization Hyperparameters

An unconstrained decision tree will continue splitting until **every leaf is completely pure** (100% training accuracy). This causes severe **overfitting (High Variance)**, fitting noise and outliers.

```
Depth 1-2 (Underfitting / High Bias)  -->  Depth 4-6 (Optimal Generalization)  -->  Unconstrained (Overfitting / High Variance)
   [ Broad coarse splits ]                      [ Meaningful patterns ]                    [ Memorizing individual noise points ]
```

### Key Regularization Hyperparameters

| Hyperparameter | Default | Description | Impact when Increased |
| :--- | :--- | :--- | :--- |
| `max_depth` | `None` (unlimited) | Maximum allowed depth of the tree from root to leaf. | **Increases** complexity (leads to overfitting if too high). |
| `min_samples_split` | `2` | Minimum number of samples a node must contain to consider a split. | **Decreases** complexity (regularizes). |
| `min_samples_leaf` | `1` | Minimum number of samples required to exist in any leaf node. | **Decreases** complexity (strongly suppresses noise fitting). |
| `max_leaf_nodes` | `None` | Caps the total number of leaves in the tree in best-first fashion. | **Increases** capacity up to the cap. |
| `min_impurity_decrease`| `0.0` | Node splits only if impurity decrease $\ge$ this threshold. | **Decreases** complexity (prunes negligible improvements). |
| `max_features` | `None` (all) | Number of features considered at each split. | Reduces variance; critical for Random Forests. |

---

## 6. Post-Pruning: Minimal Cost-Complexity Pruning ($ccp\_alpha$)

Pre-pruning limits tree growth early, but may halt before finding complex multi-feature interactions.
**Post-pruning** grows an unconstrained tree first, then prunes away branches that provide minimal statistical value.

CART uses **Cost-Complexity Pruning**:

$$R_\alpha(T) = R(T) + \alpha |T|$$

where:
- $R(T)$ is the total training error/impurity of tree $T$.
- $|T|$ is the number of terminal leaf nodes.
- $\alpha \ge 0$ (`ccp_alpha`) is the complexity parameter penalizing tree size.

### Using `cost_complexity_pruning_path`:
Scikit-Learn computes the effective alphas that prune the tree step-by-step:
```python
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
```
You can train a tree for each candidate $\alpha$ and pick the one maximizing cross-validation or validation set accuracy.

---

## 7. Feature Importance (Mean Decrease in Impurity / MDI)

Decision trees provide built-in interpretability through feature importances:

$$\text{Importance}(j) = \sum_{\text{nodes } m \text{ splitting on } j} \frac{N_m}{N_{\text{total}}} \left[ I(m) - \frac{N_{\text{left}}}{N_m} I(D_{\text{left}}) - \frac{N_{\text{right}}}{N_m} I(D_{\text{right}}) \right]$$

These values are normalized so that $\sum_{j=1}^D \text{Importance}(j) = 1.0$.

> [!WARNING]
> **Gotchas of MDI Feature Importance:**
> 1. **Cardiality Bias:** Features with many unique numeric values or high-cardinality categories have more split opportunities and artificially inflated importance.
> 2. **Correlated Features:** If two features are strongly collinear, the tree might pick one at the root and ignore the second, making the second feature appear unimportant even if it is predictive.
> 3. Use **Permutation Importance** (`sklearn.inspection.permutation_importance`) on test data for unbiased verification.

---

## 8. Visualizing Decision Trees

Scikit-Learn provides two primary tools:

### 1. Plain Text Representation (`export_text`)
Ideal for console logging, debug scripts, and fast terminal inspection:
```python
from sklearn.tree import export_text
tree_rules = export_text(clf, feature_names=feature_names)
print(tree_rules)
```

### 2. Graphical Node Diagram (`plot_tree`)
Generates high-resolution matplotlib tree graphs:
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True, rounded=True)
plt.savefig("decision_tree.png")
```

---

## 9. Strengths & Weaknesses of Decision Trees

### Advantages (Pros)
- **White-Box Interpretability:** Decisions can be converted directly into human-readable rules.
- **No Feature Scaling Required:** Invariant to monotonic transformations of features (StandardScaler / MinMaxScaler do not alter split decisions).
- **Handles Mixed Data Types:** Naturally handles continuous and binary/categorical inputs.
- **Captures Non-Linear Relationships:** Models complex non-linear boundaries and feature interactions without manual polynomial expansion.

### Disadvantages (Cons)
- **High Variance & Overfitting:** Unconstrained trees easily memorize sample noise.
- **Axis-Aligned Decision Boundaries:** Splits are always orthogonal to feature axes; diagonal boundaries require deep, stair-step approximations.
- **Instability:** Small variations in training data can lead to an entirely different tree architecture.
- **Extrapolation Inability in Regression:** Tree regressors predict constant values in each leaf; they cannot extrapolate outside the range $[y_{\min}, y_{\max}]$ of training data.
