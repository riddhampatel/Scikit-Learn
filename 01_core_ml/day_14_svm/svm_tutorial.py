"""
=============================================================================
 DAY 14 TUTORIAL: Support Vector Machines (SVM) in Scikit-Learn
=============================================================================
Topics Covered:
  1. Why Feature Scaling is Mandatory for SVM (Scaled vs Unscaled benchmark)
  2. Linear SVC & Inspecting Support Vectors (support_, support_vectors_)
  3. Non-linear Kernels (Linear, Polynomial, RBF, Sigmoid)
  4. Hyperparameter Deep Dive: Regularization Parameter 'C'
  5. Hyperparameter Deep Dive: Kernel Coefficient 'gamma'
  6. Support Vector Regression (SVR & the epsilon-tube)
  7. Multiclass Classification with SVC (OvO vs OvR)
=============================================================================
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


# =============================================================================
# 1. WHY FEATURE SCALING IS MANDATORY FOR SVM
# =============================================================================
print_section("1. Impact of Feature Scaling on SVM")

cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X_cancer, y_cancer, test_size=0.25, random_state=42, stratify=y_cancer
)

# 1A. SVM on Unscaled Data
svm_unscaled = SVC(kernel="rbf", random_state=42)
svm_unscaled.fit(X_train, y_train)
acc_unscaled = accuracy_score(y_test, svm_unscaled.predict(X_test))

# 1B. SVM on Scaled Data (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_scaled = SVC(kernel="rbf", random_state=42)
svm_scaled.fit(X_train_scaled, y_train)
acc_scaled = accuracy_score(y_test, svm_scaled.predict(X_test_scaled))

print(f"Test Accuracy WITHOUT Feature Scaling: {acc_unscaled:.4f}")
print(f"Test Accuracy WITH StandardScaler:      {acc_scaled:.4f}")
print("-> SVM relies on geometric distance calculations; scaling features prevents")
print("   large-magnitude columns from dominating the optimization objective.")


# =============================================================================
# 2. LINEAR SVC & INSPECTING SUPPORT VECTORS
# =============================================================================
print_section("2. Linear SVC & Support Vectors Inspection")

linear_svc = SVC(kernel="linear", C=1.0, random_state=42)
linear_svc.fit(X_train_scaled, y_train)
y_pred_linear = linear_svc.predict(X_test_scaled)

print(f"Linear SVC Test Accuracy: {accuracy_score(y_test, y_pred_linear):.4f}")
print(f"Total Training Samples:   {X_train_scaled.shape[0]}")
print(f"Total Support Vectors:    {len(linear_svc.support_vectors_)}")
print(f"Support Vectors per Class ({cancer.target_names[0]} vs {cancer.target_names[1]}): {linear_svc.n_support_}")
print(f"Indices of first 5 support vectors: {linear_svc.support_[:5]}")
print("-> Only these support vectors dictate the decision boundary position!")


# =============================================================================
# 3. EXPLORING KERNELS (Linear, Polynomial, RBF, Sigmoid)
# =============================================================================
print_section("3. Kernel Comparison on Non-Linear Data (make_moons)")

# Generate non-linearly separable dataset
X_moons, y_moons = make_moons(n_samples=500, noise=0.25, random_state=42)
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
    X_moons, y_moons, test_size=0.3, random_state=42, stratify=y_moons
)

scaler_m = StandardScaler()
X_m_train_scaled = scaler_m.fit_transform(X_m_train)
X_m_test_scaled = scaler_m.transform(X_m_test)

kernels = [
    ("Linear", SVC(kernel="linear", C=1.0)),
    ("Polynomial (deg=3)", SVC(kernel="poly", degree=3, C=1.0)),
    ("RBF (Gaussian)", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ("Sigmoid", SVC(kernel="sigmoid", C=1.0, gamma="scale")),
]

for name, model in kernels:
    model.fit(X_m_train_scaled, y_m_train)
    train_acc = accuracy_score(y_m_train, model.predict(X_m_train_scaled))
    test_acc = accuracy_score(y_m_test, model.predict(X_m_test_scaled))
    print(f"{name:<22} | Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f} | Num Support Vectors: {len(model.support_)}")


# =============================================================================
# 4. HYPERPARAMETER DEEP DIVE: 'C' (REGULARIZATION)
# =============================================================================
print_section("4. Impact of Regularization Parameter 'C'")

c_candidates = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

for c in c_candidates:
    clf = SVC(kernel="rbf", C=c, gamma="scale", random_state=42)
    clf.fit(X_train_scaled, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    print(f"C = {c:<7} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Support Vectors: {len(clf.support_)}")

print("-> Small C: Wider margin, more violations tolerated (Lower variance, higher bias).")
print("-> Large C: Narrower margin, heavily penalizes violations (Higher variance, lower bias).")


# =============================================================================
# 5. HYPERPARAMETER DEEP DIVE: 'gamma' (RADIUS OF INFLUENCE)
# =============================================================================
print_section("5. Impact of Kernel Coefficient 'gamma'")

gamma_candidates = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0]

for g in gamma_candidates:
    clf = SVC(kernel="rbf", C=1.0, gamma=g, random_state=42)
    clf.fit(X_train_scaled, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    print(f"gamma = {g:<6} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Support Vectors: {len(clf.support_)}")

print("-> Small gamma: Wide bell curve, support vectors have broad reach (Smooth boundary).")
print("-> Large gamma: Tight bell curve, support vectors have localized reach (Wiggly/overfitting boundary).")


# =============================================================================
# 6. SUPPORT VECTOR REGRESSION (SVR & the epsilon-tube)
# =============================================================================
print_section("6. Support Vector Regression (SVR)")

diabetes = load_diabetes()
X_diab = diabetes.data
y_diab = diabetes.target

X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(
    X_diab, y_diab, test_size=0.2, random_state=42
)

scaler_d_x = StandardScaler()
X_d_train_scaled = scaler_d_x.fit_transform(X_d_train)
X_d_test_scaled = scaler_d_x.transform(X_d_test)

# Experimenting with epsilon
epsilons = [0.1, 1.0, 10.0, 25.0]

for eps in epsilons:
    svr = SVR(kernel="rbf", C=100.0, epsilon=eps)
    svr.fit(X_d_train_scaled, y_d_train)
    y_pred_d = svr.predict(X_d_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_d_test, y_pred_d))
    r2 = r2_score(y_d_test, y_pred_d)
    print(f"epsilon = {eps:<5} | Test RMSE: {rmse:.2f} | Test R²: {r2:.4f} | Support Vectors: {len(svr.support_)}")


# =============================================================================
# 7. MULTICLASS CLASSIFICATION WITH SVC (OvO vs OvR)
# =============================================================================
print_section("7. Multiclass Classification on Wine Dataset")

wine = load_wine()
X_w, y_w = wine.data, wine.target

X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(
    X_w, y_w, test_size=0.25, random_state=42, stratify=y_w
)

scaler_w = StandardScaler()
X_w_train_scaled = scaler_w.fit_transform(X_w_train)
X_w_test_scaled = scaler_w.transform(X_w_test)

# One-vs-One (OvO) vs One-vs-Rest (OvR)
svc_ovo = SVC(kernel="rbf", decision_function_shape="ovo", random_state=42)
svc_ovo.fit(X_w_train_scaled, y_w_train)

svc_ovr = SVC(kernel="rbf", decision_function_shape="ovr", random_state=42)
svc_ovr.fit(X_w_train_scaled, y_w_train)

print(f"One-vs-One (OvO) Test Accuracy: {accuracy_score(y_w_test, svc_ovo.predict(X_w_test_scaled)):.4f}")
print(f"One-vs-Rest (OvR) Test Accuracy: {accuracy_score(y_w_test, svc_ovr.predict(X_w_test_scaled)):.4f}")
print(f"Number of classes: {len(np.unique(y_w))}")
print(f"OvO builds K*(K-1)/2 = 3*(2)/2 = 3 binary classifiers.")
print("\nClassification Report (OvR):")
print(classification_report(y_w_test, svc_ovr.predict(X_w_test_scaled), target_names=wine.target_names))
