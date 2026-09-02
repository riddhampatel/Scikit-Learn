"""
Day 13: Naive Bayes Classifiers
================================
A hands-on guide covering:
1. GaussianNB on Continuous Data (Breast Cancer Dataset)
2. MultinomialNB for Text / Count Data (Spam vs Ham Classification)
3. BernoulliNB for Binary Feature Indicators
4. ComplementNB for Imbalanced Class Distributions
5. Laplace Smoothing (alpha parameter tuning)
6. Extracting Top Informative Features from Log Likelihoods
"""

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    BernoulliNB,
    ComplementNB
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =====================================================================
# SECTION 1: GaussianNB on Continuous Data
# =====================================================================
print("=" * 70)
print("SECTION 1: GaussianNB on Continuous Features (Breast Cancer)")
print("=" * 70)

cancer = load_breast_cancer()
X_cancer, y_cancer = cancer.data, cancer.target
target_names = cancer.target_names  # ['malignant', 'benign']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
)

# Initialize and fit GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_c, y_train_c)

# Evaluate predictions
y_pred_c = gnb.predict(X_test_c)
y_proba_c = gnb.predict_proba(X_test_c)

print(f"Dataset shape: {X_cancer.shape}")
print(f"GaussianNB Test Accuracy: {accuracy_score(y_test_c, y_pred_c):.4f}")
print("\nClass Priors learned P(y):")
for cls_idx, prior in enumerate(gnb.class_prior_):
    print(f"  Class '{target_names[cls_idx]}': {prior:.4f}")

print("\nMean (theta_) of first 3 features per class:")
for cls_idx, mean_vec in enumerate(gnb.theta_):
    print(f"  Class '{target_names[cls_idx]}': {mean_vec[:3].round(3)}")

print("\nSample Probabilities (predict_proba) for first 3 test samples:")
for i in range(3):
    print(f"  Sample {i+1} [P(Malignant)={y_proba_c[i, 0]:.4f}, P(Benign)={y_proba_c[i, 1]:.4f}] -> Pred: {target_names[y_pred_c[i]]} (True: {target_names[y_test_c[i]]})")


# =====================================================================
# SECTION 2: MultinomialNB for Text Classification (Spam vs Ham)
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 2: MultinomialNB on Text Data (Spam vs Ham)")
print("=" * 70)

# Sample corpus of SMS / Email messages
corpus = [
    # Ham (Legitimate) - 0
    "Hey are we still meeting for lunch today at twelve?",
    "Please find attached the quarterly project budget report for review.",
    "Can you send me the lecture slides from yesterday morning?",
    "Let's schedule a team sync tomorrow at 10am to discuss project roadmap.",
    "Great work on the presentation yesterday, client was very impressed!",
    "Are you coming to dinner with mom and dad this weekend?",
    "Don't forget to push your latest commits to the git repository.",
    "The meeting has been rescheduled to 3pm in conference room B.",
    "Happy birthday! Wishing you a wonderful year ahead.",
    "Thanks for the quick response, I'll review the pull request shortly.",
    
    # Spam - 1
    "CONGRATULATIONS! You have won a $1,000 Walmart gift card! Claim now at freegift.com",
    "URGENT: Your bank account is locked. Click here to verify your identity immediately!",
    "Exclusive loan offer: get $50,000 cash loan with instant approval today!",
    "Win a brand new iPhone 16 Pro! Click this link to claim your reward instantly.",
    "Act now! Limited time secret discount: 90% off luxury designer watches!",
    "You have been selected for a free crypto prize giveaway. Deposit now to unlock.",
    "Urgent notification: unclaimed money waiting for you. Call this toll-free number now!",
    "Double your investment in 24 hours guaranteed! Sign up today for free trial.",
    "Earn $5,000 a week working from home! No experience required, click here.",
    "Final notice: Your subscription is expiring. Renew now to avoid extra charges."
]

labels = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 10 Ham
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1   # 10 Spam
])

X_train_txt, X_test_txt, y_train_txt, y_test_txt = train_test_split(
    corpus, labels, test_size=0.3, random_state=42, stratify=labels
)

# Convert text documents to a token count matrix (Bag of Words)
vectorizer = CountVectorizer(stop_words='english')
X_train_bow = vectorizer.fit_transform(X_train_txt)
X_test_bow = vectorizer.transform(X_test_txt)

print(f"Vocabulary Size (Unique words): {len(vectorizer.get_feature_names_out())}")

# Fit MultinomialNB
mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_bow, y_train_txt)

y_pred_mnb = mnb.predict(X_test_bow)
print(f"MultinomialNB Test Accuracy: {accuracy_score(y_test_txt, y_pred_mnb):.4f}")


# =====================================================================
# SECTION 3: Classifying New Unseen Text Samples
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 3: Inference on Unseen Custom Messages")
print("=" * 70)

new_messages = [
    "Hey, let's grab coffee and review the project code tomorrow morning.",
    "WINNER! Urgent: Claim your free luxury gift card and cash reward today!"
]

new_bow = vectorizer.transform(new_messages)
new_preds = mnb.predict(new_bow)
new_probas = mnb.predict_proba(new_bow)

for msg, pred, proba in zip(new_messages, new_preds, new_probas):
    label_str = "SPAM" if pred == 1 else "HAM"
    print(f"Text: \"{msg}\"")
    print(f"  -> Prediction: {label_str} [P(Ham)={proba[0]:.3f}, P(Spam)={proba[1]:.3f}]\n")


# =====================================================================
# SECTION 4: BernoulliNB vs MultinomialNB vs ComplementNB
# =====================================================================
print("=" * 70)
print("SECTION 4: Comparing Naive Bayes Text Variants")
print("=" * 70)

# BernoulliNB (binarized presence / absence)
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train_bow, y_train_txt)
y_pred_bnb = bnb.predict(X_test_bow)

# ComplementNB (tailored for imbalanced distributions)
cnb = ComplementNB(alpha=1.0)
cnb.fit(X_train_bow, y_train_txt)
y_pred_cnb = cnb.predict(X_test_bow)

print(f"MultinomialNB Accuracy: {accuracy_score(y_test_txt, y_pred_mnb):.4f}")
print(f"BernoulliNB   Accuracy: {accuracy_score(y_test_txt, y_pred_bnb):.4f}")
print(f"ComplementNB  Accuracy: {accuracy_score(y_test_txt, y_pred_cnb):.4f}")


# =====================================================================
# SECTION 5: Effect of Laplace Smoothing Hyperparameter (alpha)
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 5: Laplace Smoothing (alpha parameter tuning)")
print("=" * 70)

alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

for a in alphas:
    clf = MultinomialNB(alpha=a)
    clf.fit(X_train_bow, y_train_txt)
    train_score = clf.score(X_train_bow, y_train_txt)
    test_score = clf.score(X_test_bow, y_test_txt)
    print(f"alpha = {a:<6} | Train Acc: {train_score:.4f} | Test Acc: {test_score:.4f}")


# =====================================================================
# SECTION 6: Inspecting Most Informative Words (Feature Log Probs)
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 6: Top Indicative Words by Class Log Likelihood")
print("=" * 70)

feature_names = np.array(vectorizer.get_feature_names_out())

# feature_log_prob_ shape: (n_classes, n_features)
# Log probability P(word | class)
for cls_idx, cls_name in enumerate(["HAM", "SPAM"]):
    log_probs = mnb.feature_log_prob_[cls_idx]
    # Sort indices by highest probability
    top_indices = np.argsort(log_probs)[::-1][:5]
    top_words = feature_names[top_indices]
    print(f"Top words indicative of {cls_name}: {list(top_words)}")

print("\n" + "=" * 70)
print("Tutorial complete! Run 'python challenge.py' to test your active recall.")
print("=" * 70)
