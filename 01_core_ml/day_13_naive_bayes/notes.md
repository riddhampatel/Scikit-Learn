# 📝 Day 13 — Naive Bayes Classifier

## What is Naive Bayes?
**Naive Bayes** is a family of fast, probabilistic supervised machine learning algorithms based on **Bayes' Theorem**. It is primarily used for **classification** problems (both binary and multiclass), especially in **natural language processing (NLP)**, **spam detection**, and **sentiment analysis**.

Despite its simplicity and strong theoretical assumptions, Naive Bayes often performs remarkably well, trains in $O(N \cdot D)$ time (linear in samples and features), and requires relatively small amounts of training data.

---

## The Mathematical Foundation: Bayes' Theorem

Bayes' Theorem describes the probability of an event (class $y$) based on prior knowledge of conditions that might be related to the event (features $\mathbf{X} = (x_1, x_2, \dots, x_n)$):

$$P(y \mid \mathbf{X}) = \frac{P(\mathbf{X} \mid y) \cdot P(y)}{P(\mathbf{X})}$$

### Key Components:
1. **Posterior Probability $P(y \mid \mathbf{X})$:** Probability that sample $\mathbf{X}$ belongs to class $y$, given its observed features.
2. **Prior Probability $P(y)$:** Baseline probability of class $y$ occurring in the dataset (class frequency).
3. **Likelihood $P(\mathbf{X} \mid y)$:** Probability of observing features $\mathbf{X}$ given that the class is $y$.
4. **Evidence / Marginal Probability $P(\mathbf{X})$:** Total probability of observing features $\mathbf{X}$ across all classes. Since $P(\mathbf{X})$ is constant for all candidate classes, it acts as a normalizer and can be dropped when comparing classes:

$$P(y \mid \mathbf{X}) \propto P(\mathbf{X} \mid y) \cdot P(y)$$

---

## Why is it Called "Naive"?

In real-world data, features often correlate with each other (e.g., in text, the word *"machine"* frequently co-occurs with *"learning"*).

Naive Bayes makes the **"naive" conditional independence assumption**:
> It assumes that all features $x_1, x_2, \dots, x_n$ are **mutually independent** given the class label $y$.

Under this assumption, the joint likelihood factors into the product of individual feature likelihoods:

$$P(\mathbf{X} \mid y) = P(x_1, x_2, \dots, x_n \mid y) = \prod_{i=1}^{n} P(x_i \mid y)$$

Thus, the classification decision rule (Maximum A Posteriori or MAP) is:

$$\hat{y} = \arg\max_y \left( P(y) \prod_{i=1}^{n} P(x_i \mid y) \right)$$

---

## Log Probabilities to Prevent Underflow

Multiplying hundreds or thousands of small probabilities ($p < 1.0$) causes floating-point **numerical underflow** (rounding down to zero). 

To prevent this, Naive Bayes computes the **log-likelihood sum**:

$$\log P(y \mid \mathbf{X}) \propto \log P(y) + \sum_{i=1}^{n} \log P(x_i \mid y)$$

Since logarithm is a monotonically increasing function, finding the $\arg\max$ of the log-sum yields the exact same class prediction.

---

## Variants of Naive Bayes in Scikit-Learn

The choice of Naive Bayes variant depends on the **nature and distribution of your feature data**:

```
                              Feature Types
                                    |
     +------------------------------+------------------------------+
     |                              |                              |
Continuous / Real-Valued       Word / Discrete Counts          Binary (0 or 1)
     |                              |                              |
GaussianNB                    MultinomialNB / ComplementNB    BernoulliNB
```

### 1. `GaussianNB` (Continuous Features)
- **Assumption:** Continuous features follow a normal (Gaussian) distribution within each class:
  $$P(x_i \mid y) = \frac{1}{\sqrt{2\pi \sigma_{y,i}^2}} \exp\left(-\frac{(x_i - \mu_{y,i})^2}{2\sigma_{y,i}^2}\right)$$
- **Parameters learned:** Mean $\mu_{y,i}$ and variance $\sigma_{y,i}^2$ for each feature $i$ in each class $y$.
- **Best for:** Tabular datasets with continuous numeric features (e.g., Iris, Wine, Breast Cancer).
- **Note:** Does not require heavy feature scaling, though non-skewed Gaussian distributions yield the best results.

### 2. `MultinomialNB` (Count / Frequency Features)
- **Assumption:** Features represent discrete counts or integer frequencies (e.g., word occurrence count in a document):
  $$P(x_i \mid y) = \theta_{y,i} = \frac{N_{yi} + \alpha}{N_y + \alpha n}$$
- **Best for:** Text classification with Bag-of-Words (`CountVectorizer`) or TF-IDF (`TfidfVectorizer`).
- **Requirement:** Features must be non-negative ($x \ge 0$).

### 3. `BernoulliNB` (Binary / Boolean Features)
- **Assumption:** Features are binary variables ($x_i \in \{0, 1\}$) representing the presence or absence of a feature:
  $$P(x_i \mid y) = P(i \mid y) x_i + (1 - P(i \mid y))(1 - x_i)$$
- **Best for:** Short text classification, keyword boolean presence (e.g., "does word appear in email: Yes/No").
- **Key difference from MultinomialNB:** BernoulliNB explicitly penalizes the *absence* of words indicative of a class.

### 4. `ComplementNB` (Imbalanced Text Datasets)
- **Design:** Adapts MultinomialNB for **imbalanced datasets** by estimating parameters using statistics from all classes *except* class $y$ (the complement of class $y$).
- **Best for:** Highly skewed/imbalanced multiclass text classification benchmarks.

### 5. `CategoricalNB` (Categorical Features)
- **Assumption:** Features are discrete categorical values (e.g., color: Red=0, Blue=1, Green=2).

---

## The Zero-Frequency Problem & Laplace Smoothing (`alpha`)

### The Problem:
If a word (e.g., *"cryptocurrency"*) appears in test data under class *"Spam"*, but never appeared in the training samples for *"Spam"*, then:
$$P(\text{"cryptocurrency"} \mid \text{Spam}) = 0$$

Because likelihoods are multiplied:
$$\prod_{i=1}^{n} P(x_i \mid \text{Spam}) = 0$$

A single unseen feature completely zeroes out the entire class probability!

### The Solution: Additive (Laplace) Smoothing
Laplace smoothing adds a pseudo-count $\alpha > 0$ to feature frequencies:

$$\hat{\theta}_{yi} = \frac{N_{yi} + \alpha}{N_y + \alpha \cdot D}$$

- $\alpha = 1.0$ $\implies$ **Laplace Smoothing** (default in Scikit-Learn).
- $0 < \alpha < 1.0$ $\implies$ **Lidstone Smoothing**.
- $\alpha = 0$ $\implies$ No smoothing (risky).

---

## Summary of Scikit-Learn API

| Estimator | Key Hyperparameters | Key Attributes | Ideal Use Case |
| :--- | :--- | :--- | :--- |
| `GaussianNB` | `var_smoothing=1e-9` | `class_prior_`, `theta_` (means), `var_` (variances) | Continuous tabular features |
| `MultinomialNB` | `alpha=1.0`, `fit_prior=True` | `class_log_prior_`, `feature_log_prob_` | Word counts / TF-IDF text data |
| `BernoulliNB` | `alpha=1.0`, `binarize=0.0` | `class_log_prior_`, `feature_log_prob_` | Binary feature indicators (0/1) |
| `ComplementNB`| `alpha=1.0`, `norm=False` | `feature_log_prob_` | Imbalanced text datasets |

---

## Strengths vs Limitations

### ✅ Advantages:
1. **Blazing Fast:** Extremely fast training and inference — $O(N \cdot D)$.
2. **Scales to High Dimensions:** Works exceptionally well even when $D \gg N$ (e.g., 20,000 text vocab terms with 1,000 documents).
3. **Resistant to Overfitting:** Few parameters to tune; minimal risk of variance explosion.
4. **Handles Missing/Irrelevant Data:** If a feature isn't informative, its distribution is similar across classes.
5. **Great Baseline:** Often the first classification model to benchmark on NLP tasks.

### ❌ Limitations:
1. **Independence Assumption Violation:** Strongly correlated features can bias probability estimates (though classification decisions often remain surprisingly robust).
2. **Probability Calibration:** Output probabilities (`predict_proba`) are often pushed towards extremes ($0.0$ or $1.0$) and may not be well-calibrated confidence scores.
3. **Continuous Data Assumption:** GaussianNB assumes normality, which may fail on multimodal or heavy-tailed distributions.
