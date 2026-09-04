"""
DAY 15 ACTIVE RECALL CHALLENGE: Decision Trees
"""

import numpy as np

from sklearn.datasets import load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score
)


# ===================================================================
# TASK 1: Load Wine Dataset & Stratified Train/Test Split
# ===================================================================
# 1. Load the Wine dataset using `load_wine()`.
# 2. Extract feature matrix X and target array y.
# 3. Perform a stratified train/test split:
#    - test_size = 0.25
#    - stratify = y
#    - random_state = 42

wine = load_wine()

# Extract feature matrix X and target labels y
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

# Perform stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)


# ===================================================================
# TASK 2: Baseline Unconstrained Tree (Overfitting Diagnostic)
# ===================================================================
# 1. Initialize an unconstrained `DecisionTreeClassifier(random_state=42)`.
# 2. Fit the classifier on `X_train` and `y_train`.
# 3. Predict on both train set and test set.
# 4. Extract tree depth (`get_depth()`) and number of leaves (`get_n_leaves()`).
# 5. Calculate train accuracy and test accuracy.

# Initialize and fit unconstrained DecisionTreeClassifier
unconstrained_clf = DecisionTreeClassifier(random_state=42)

unconstrained_clf.fit(X_train, y_train)

# Predict on train and test sets
y_train_pred_unconstrained = unconstrained_clf.predict(X_train)
y_test_pred_unconstrained = unconstrained_clf.predict(X_test)

# Extract depth and leaf count
tree_depth = unconstrained_clf.get_depth()
n_leaves = unconstrained_clf.get_n_leaves()

# Compute train and test accuracy
train_acc_unconstrained = accuracy_score(
    y_train,
    y_train_pred_unconstrained
)

test_acc_unconstrained = accuracy_score(
    y_test,
    y_test_pred_unconstrained
)

print("=" * 60)
print("TASK 2: UNCONSTRAINED DECISION TREE")
print("=" * 60)
print(f"Tree Depth:        {tree_depth}")
print(f"Leaf Count:        {n_leaves}")
print(f"Train Accuracy:    {train_acc_unconstrained}")
print(f"Test Accuracy:     {test_acc_unconstrained}")


# ===================================================================
# TASK 3: Regularized Tree with Pre-Pruning Hyperparameters
# ===================================================================
# 1. Initialize `DecisionTreeClassifier` with pre-pruning:
#    - max_depth = 3
#    - min_samples_leaf = 3
#    - min_samples_split = 4
#    - random_state = 42
# 2. Fit on `X_train` and `y_train`.
# 3. Predict on test set.
# 4. Compute test accuracy and weighted F1-score.

# Initialize and fit regularized DecisionTreeClassifier
regularized_clf = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=3,
    min_samples_split=4,
    random_state=42
)

regularized_clf.fit(X_train, y_train)

# Predict on test set
y_test_pred_reg = regularized_clf.predict(X_test)

# Compute test accuracy and weighted F1-score
test_acc_reg = accuracy_score(
    y_test,
    y_test_pred_reg
)

test_f1_reg = f1_score(
    y_test,
    y_test_pred_reg,
    average="weighted"
)

print("\n" + "=" * 60)
print("TASK 3: REGULARIZED DECISION TREE")
print("=" * 60)
print(f"Test Accuracy:     {test_acc_reg}")
print(f"Test F1-Score:     {test_f1_reg}")


# ===================================================================
# TASK 4: Split Criteria Comparison (Gini vs Entropy)
# ===================================================================
# 1. Train model with criterion="gini" (max_depth=3, random_state=42).
# 2. Train model with criterion="entropy" (max_depth=3, random_state=42).
# 3. Compare test accuracies of both models.

# Train Gini tree
gini_clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

gini_clf.fit(X_train, y_train)

gini_pred = gini_clf.predict(X_test)

gini_test_acc = accuracy_score(
    y_test,
    gini_pred
)

# Train Entropy tree
entropy_clf = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)

entropy_clf.fit(X_train, y_train)

entropy_pred = entropy_clf.predict(X_test)

entropy_test_acc = accuracy_score(
    y_test,
    entropy_pred
)

print("\n" + "=" * 60)
print("TASK 4: GINI VS ENTROPY SPLIT CRITERIA")
print("=" * 60)
print(f"Gini Test Accuracy:    {gini_test_acc}")
print(f"Entropy Test Accuracy: {entropy_test_acc}")


# ===================================================================
# TASK 5: Feature Importances Extraction & Top 5 Ranking
# ===================================================================
# 1. Extract feature importances from `regularized_clf.feature_importances_`.
# 2. Identify the indices of the top 5 most important features.
# 3. Store a list of tuples: (feature_name, importance_score) for the top 5.

# Extract feature importances
importances = regularized_clf.feature_importances_

# Get top 5 feature indices
top_5_indices = np.argsort(importances)[::-1][:5]

# Create list of (feature_name, importance_score)
top_5_features = [
    (feature_names[i], importances[i])
    for i in top_5_indices
]

print("\n" + "=" * 60)
print("TASK 5: TOP 5 FEATURE IMPORTANCES (MDI)")
print("=" * 60)

if top_5_features is not None:
    for rank, (name, score) in enumerate(top_5_features, start=1):
        print(f"#{rank} {name:<30}: {score:.4f}")
else:
    print("Top 5 features: None")


# ===================================================================
# TASK 6: Post-Pruning with Cost-Complexity Pruning (ccp_alpha)
# ===================================================================
# 1. Compute the cost complexity pruning path on `unconstrained_clf`
#    using `cost_complexity_pruning_path(X_train, y_train)`.
# 2. Extract `ccp_alphas` from the pruning path.
# 3. Train a DecisionTreeClassifier for each alpha (excluding the last one).
# 4. Find the alpha that maximizes test accuracy.

# Extract ccp_alphas from unconstrained_clf
path = unconstrained_clf.cost_complexity_pruning_path(
    X_train,
    y_train
)

ccp_alphas = path.ccp_alphas

# Find best ccp_alpha and highest test accuracy
best_alpha = None
best_pruned_acc = -1

for alpha in ccp_alphas[:-1]:

    clf = DecisionTreeClassifier(
        random_state=42,
        ccp_alpha=alpha
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(
        y_test,
        y_pred
    )

    if acc > best_pruned_acc:
        best_pruned_acc = acc
        best_alpha = alpha

print("\n" + "=" * 60)
print("TASK 6: COST-COMPLEXITY PRUNING (ccp_alpha)")
print("=" * 60)
print(f"Best ccp_alpha:    {best_alpha}")
print(f"Best Pruned Acc:   {best_pruned_acc}")


# ===================================================================
# TASK 7: Decision Tree Regression (DecisionTreeRegressor)
# ===================================================================
# 1. Load the diabetes dataset using `load_diabetes()`.
# 2. Split into train and test sets (test_size=0.2, random_state=42).
# 3. Initialize and fit `DecisionTreeRegressor(max_depth=3, random_state=42)`.
# 4. Predict on test set and calculate RMSE and R² score.

diabetes = load_diabetes()

X_reg = diabetes.data
y_reg = diabetes.target

# Train/test split for regression
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

# Initialize and fit DecisionTreeRegressor
tree_regressor = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

tree_regressor.fit(
    X_train_r,
    y_train_r
)

# Predict on test set
y_pred_r = tree_regressor.predict(X_test_r)

# Calculate test RMSE and R² score
rmse_r = np.sqrt(
    mean_squared_error(
        y_test_r,
        y_pred_r
    )
)

r2_r = r2_score(
    y_test_r,
    y_pred_r
)

print("\n" + "=" * 60)
print("TASK 7: DECISION TREE REGRESSION")
print("=" * 60)
print(f"Test RMSE:         {rmse_r}")
print(f"Test R² Score:     {r2_r}")