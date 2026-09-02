"""
DAY 13 ACTIVE RECALL CHALLENGE: Naive Bayes Classifier
"""

import numpy as np

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ===================================================================
# TASK 1: Continuous Data - Dataset Load & Split
# ===================================================================
# 1. Load the Wine dataset using `load_wine()`.
# 2. Extract feature matrix X and target labels y.
# 3. Split into train and test sets (test_size=0.2, stratify=y, random_state=42).

data = load_wine()

X = data.data
y = data.target

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ===================================================================
# TASK 2: Continuous Data - Fit GaussianNB & Evaluate
# ===================================================================
# 1. Initialize `GaussianNB()`.
# 2. Fit on training data.
# 3. Predict class labels.
# 4. Compute accuracy and classification report.

gnb_model = GaussianNB()

gnb_model.fit(X_train_c, y_train_c)

y_pred_gnb = gnb_model.predict(X_test_c)

accuracy_gnb = accuracy_score(y_test_c, y_pred_gnb)

print("=" * 60)
print("GAUSSIAN NAIVE BAYES (CONTINUOUS FEATURES)")
print("=" * 60)
print(f"Test Accuracy: {accuracy_gnb:.4f}")

print(
    "\nClassification Report:\n",
    classification_report(
        y_test_c,
        y_pred_gnb,
        target_names=data.target_names
    )
)


# ===================================================================
# TASK 3: Text Data - Vectorization (Bag-of-Words)
# ===================================================================

corpus = [
    # Class 0: Tech / Programming
    "Python is an interpreted high-level general-purpose programming language.",
    "Machine learning algorithms build a mathematical model based on sample data.",
    "Deep learning models utilize artificial neural networks with multiple layers.",
    "Git is a distributed version control system for tracking changes in source code.",
    "Refactoring code improves maintainability and readable software architecture.",
    "Scikit-learn provides simple and efficient tools for predictive data analysis.",
    "Database indexing significantly accelerates SQL query performance.",
    "Modern cloud computing leverages containerization and microservices architecture.",
    
    # Class 1: Cooking / Food
    "Preheat oven to 375 degrees and bake the chocolate cookies for 12 minutes.",
    "Heat olive oil in a skillet and saute fresh garlic and diced onions.",
    "Whisk flour, eggs, butter, and milk together until smooth batter forms.",
    "Simmer tomato sauce with basil, oregano, and crushed red pepper flakes.",
    "Season grilled salmon fillet with kosher salt, black pepper, and lemon juice.",
    "Boil pasta in salted water until al dente then drain and toss in sauce.",
    "Roast seasoned vegetables on a baking sheet until tender and caramelized.",
    "Chop fresh herbs and mix with balsamic vinegar for salad dressing."
]

labels = np.array([
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1
])


# Train/test split
X_train_txt, X_test_txt, y_train_txt, y_test_txt = train_test_split(
    corpus,
    labels,
    test_size=0.3,
    stratify=labels,
    random_state=42
)


# Initialize CountVectorizer
vectorizer = CountVectorizer(stop_words="english")


# Fit and transform training data
X_train_bow = vectorizer.fit_transform(X_train_txt)


# Transform test data
X_test_bow = vectorizer.transform(X_test_txt)


# ===================================================================
# TASK 4: Text Data - Train MultinomialNB & ComplementNB
# ===================================================================
# 1. Initialize MultinomialNB(alpha=1.0)
# 2. Fit and predict
# 3. Initialize ComplementNB(alpha=1.0)
# 4. Fit and predict

# -------------------------------
# Multinomial Naive Bayes
# -------------------------------

mnb_model = MultinomialNB(alpha=1.0)

mnb_model.fit(X_train_bow, y_train_txt)

y_pred_mnb = mnb_model.predict(X_test_bow)

accuracy_mnb = accuracy_score(y_test_txt, y_pred_mnb)


# -------------------------------
# Complement Naive Bayes
# -------------------------------

cnb_model = ComplementNB(alpha=1.0)

cnb_model.fit(X_train_bow, y_train_txt)

y_pred_cnb = cnb_model.predict(X_test_bow)

accuracy_cnb = accuracy_score(y_test_txt, y_pred_cnb)


print("\n" + "=" * 60)
print("TEXT CLASSIFICATION RESULTS")
print("=" * 60)
print(f"MultinomialNB Accuracy: {accuracy_mnb:.4f}")
print(f"ComplementNB  Accuracy: {accuracy_cnb:.4f}")


# ===================================================================
# TASK 5: Laplace Smoothing (alpha parameter tuning)
# ===================================================================
# 1. Iterate over candidate alpha values.
# 2. Train MultinomialNB(alpha=a).
# 3. Record train and test scores.

alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

print("\n" + "=" * 60)
print("LAPLACE SMOOTHING (ALPHA TUNING)")
print("=" * 60)

for a in alphas:

    model = MultinomialNB(alpha=a)

    model.fit(X_train_bow, y_train_txt)

    train_accuracy = model.score(X_train_bow, y_train_txt)
    test_accuracy = model.score(X_test_bow, y_test_txt)

    print(
        f"Alpha: {a:<5} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Test Accuracy: {test_accuracy:.4f}"
    )