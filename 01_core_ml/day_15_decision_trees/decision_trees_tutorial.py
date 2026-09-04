"""
=============================================================================
 DAY 15 TUTORIAL: Decision Trees in Scikit-Learn
=============================================================================
Topics Covered:
  1. Classification with DecisionTreeClassifier: Unconstrained vs Regularized
  2. Split Criteria: Gini Impurity vs Shannon Entropy (Information Gain)
  3. Hyperparameter Deep Dive: max_depth, min_samples_split, min_samples_leaf
  4. Tree Visualization: Text Rules (export_text) & Graphical (plot_tree)
  5. Feature Importance Analysis (Mean Decrease in Impurity / MDI)
  6. Post-Pruning via Cost-Complexity Pruning (ccp_alpha path)
  7. Regression with DecisionTreeRegressor (Step function approximation)
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    export_text,
    plot_tree
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score
)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


# =============================================================================
# 1. CLASSIFICATION: UNCONSTRAINED VS REGULARIZED TREES (OVERFITTING DEMO)
# =============================================================================
print_section("1. Unconstrained vs Regularized Decision Trees")

cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target
feature_names = cancer.feature_names
target_names = cancer.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X_cancer, y_cancer, test_size=0.25, random_state=42, stratify=y_cancer
)

# 1A. Fully Grown Unconstrained Tree
unconstrained_tree = DecisionTreeClassifier(random_state=42)
unconstrained_tree.fit(X_train, y_train)

train_acc_unconstrained = accuracy_score(y_train, unconstrained_tree.predict(X_train))
test_acc_unconstrained = accuracy_score(y_test, unconstrained_tree.predict(X_test))

print(f"Unconstrained Tree Depth:      {unconstrained_tree.get_depth()}")
print(f"Unconstrained Leaf Count:      {unconstrained_tree.get_n_leaves()}")
print(f"Unconstrained Train Accuracy:  {train_acc_unconstrained * 100:.2f}% (Memorization / Overfitting)")
print(f"Unconstrained Test Accuracy:   {test_acc_unconstrained * 100:.2f}%")

# 1B. Constrained Tree with Pre-Pruning (max_depth=3, min_samples_leaf=5)
regularized_tree = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)
regularized_tree.fit(X_train, y_train)

train_acc_reg = accuracy_score(y_train, regularized_tree.predict(X_train))
test_acc_reg = accuracy_score(y_test, regularized_tree.predict(X_test))

print("\nRegularized Tree (max_depth=3, min_samples_leaf=5):")
print(f"Regularized Tree Depth:        {regularized_tree.get_depth()}")
print(f"Regularized Leaf Count:        {regularized_tree.get_n_leaves()}")
print(f"Regularized Train Accuracy:    {train_acc_reg * 100:.2f}%")
print(f"Regularized Test Accuracy:     {test_acc_reg * 100:.2f}%")
print(f"Generalization Gap Reduced by: {(train_acc_unconstrained - test_acc_unconstrained) - (train_acc_reg - test_acc_reg):.4f}")


# =============================================================================
# 2. SPLIT CRITERIA: GINI IMPURITY VS SHANNON ENTROPY
# =============================================================================
print_section("2. Split Criteria: Gini Impurity vs Entropy")

tree_gini = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
tree_gini.fit(X_train, y_train)
acc_gini = accuracy_score(y_test, tree_gini.predict(X_test))

tree_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
tree_entropy.fit(X_train, y_train)
acc_entropy = accuracy_score(y_test, tree_entropy.predict(X_test))

print(f"Gini Criterion    -> Test Accuracy: {acc_gini:.4f} | Leaves: {tree_gini.get_n_leaves()}")
print(f"Entropy Criterion -> Test Accuracy: {acc_entropy:.4f} | Leaves: {tree_entropy.get_n_leaves()}")
print("-> Gini is computationally faster (no logarithms); empirical differences are usually minimal.")


# =============================================================================
# 3. HYPERPARAMETER DEEP DIVE: DEPTH SWEEP
# =============================================================================
print_section("3. Hyperparameter Deep Dive: Max Depth Sweep")

depth_range = range(1, 11)
train_scores = []
test_scores = []

for depth in depth_range:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
    test_scores.append(accuracy_score(y_test, clf.predict(X_test)))

print(f"{'Depth':<8} | {'Train Accuracy':<16} | {'Test Accuracy':<16} | {'Status'}")
print("-" * 55)
for depth, tr, te in zip(depth_range, train_scores, test_scores):
    status = "Optimal Range" if depth in [3, 4] else ("Underfitting" if depth < 3 else "Overfitting")
    print(f"{depth:<8} | {tr * 100:<15.2f}% | {te * 100:<15.2f}% | {status}")


# =============================================================================
# 4. TREE VISUALIZATION: TEXT RULES & GRAPHICAL PLOT
# =============================================================================
print_section("4. Tree Visualization: Rules and Plot")

# Train a small intuitive tree on the Iris dataset for clean visualization
iris = load_iris()
iris_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
iris_tree.fit(iris.data, iris.target)

print("Text-based Decision Tree Rules (Iris Dataset):")
tree_text = export_text(iris_tree, feature_names=list(iris.feature_names), spacing=3)
print(tree_text)

# Save graphical tree plot to file
current_dir = os.path.dirname(os.path.abspath(__file__))
plot_path = os.path.join(current_dir, "decision_tree_visualization.png")

plt.figure(figsize=(14, 8), dpi=150)
plot_tree(
    iris_tree,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Visualization (Iris Dataset, max_depth=3)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(plot_path)
plt.close()
print(f"Graphical tree plot successfully saved to: {plot_path}")


# =============================================================================
# 5. FEATURE IMPORTANCE ANALYSIS (MEAN DECREASE IN IMPURITY)
# =============================================================================
print_section("5. Feature Importances (MDI)")

importances = regularized_tree.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

print("Top 10 Most Important Features in Breast Cancer Classifier:")
print(f"{'Rank':<6} | {'Feature Name':<32} | {'Importance Score'}")
print("-" * 60)
for rank, idx in enumerate(sorted_indices[:10], start=1):
    print(f"#{rank:<5} | {feature_names[idx]:<32} | {importances[idx]:.4f}")

print(f"\nSum of all feature importances: {np.sum(importances):.4f} (always sums to 1.0)")


# =============================================================================
# 6. POST-PRUNING: COST-COMPLEXITY PRUNING (ccp_alpha)
# =============================================================================
print_section("6. Post-Pruning via Cost-Complexity Path (ccp_alpha)")

# Get effective pruning alphas from the unconstrained tree
pruning_path = unconstrained_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = pruning_path.ccp_alphas[:-1]  # Exclude maximum alpha which leaves only root

candidate_alphas = ccp_alphas[::max(1, len(ccp_alphas) // 6)]  # Sample ~6 alpha checkpoints
pruned_models = []

print(f"{'ccp_alpha':<12} | {'Depth':<8} | {'Leaves':<8} | {'Train Acc':<12} | {'Test Acc':<12}")
print("-" * 60)

best_alpha = 0.0
best_test_acc = 0.0
best_pruned_tree = None

for alpha in candidate_alphas:
    tree_alpha = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    tree_alpha.fit(X_train, y_train)
    
    tr_acc = accuracy_score(y_train, tree_alpha.predict(X_train))
    te_acc = accuracy_score(y_test, tree_alpha.predict(X_test))
    
    print(f"{alpha:<12.5f} | {tree_alpha.get_depth():<8} | {tree_alpha.get_n_leaves():<8} | {tr_acc * 100:<11.2f}% | {te_acc * 100:<11.2f}%")
    
    if te_acc > best_test_acc:
        best_test_acc = te_acc
        best_alpha = alpha
        best_pruned_tree = tree_alpha

print(f"\nOptimal ccp_alpha found: {best_alpha:.5f} with Test Accuracy: {best_test_acc * 100:.2f}%")


# =============================================================================
# 7. DECISION TREE REGRESSION (PIECEWISE STEP APPROXIMATION)
# =============================================================================
print_section("7. Decision Tree Regressor (DecisionTreeRegressor)")

diabetes = load_diabetes()
X_reg = diabetes.data
y_reg = diabetes.target

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Compare Regressor with different max_depth values
for depth in [2, 4, 8, None]:
    reg = DecisionTreeRegressor(max_depth=depth, random_state=42)
    reg.fit(X_train_r, y_train_r)
    
    y_pred_tr = reg.predict(X_train_r)
    y_pred_te = reg.predict(X_test_r)
    
    r2_tr = r2_score(y_train_r, y_pred_tr)
    r2_te = r2_score(y_test_r, y_pred_te)
    rmse_te = np.sqrt(mean_squared_error(y_test_r, y_pred_te))
    
    depth_label = str(depth) if depth is not None else "None (Unconstrained)"
    print(f"Max Depth: {depth_label:<20} | Train R²: {r2_tr:.4f} | Test R²: {r2_te:.4f} | Test RMSE: {rmse_te:.2f}")

print("\nKey Takeaway: Unconstrained tree achieves Train R² = 1.0000 but overfits (Test R² drops).")
print("Regularizing depth (e.g. depth=4) produces far superior generalization.")
