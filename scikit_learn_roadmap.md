# 🧠 Complete Scikit-learn Mastery Roadmap
### Beginner → Advanced → Machine Learning Engineer

**Duration:** 60 Days  
**Structure:** Theory → Hands-on Tutorial → Active Recall Challenge → Progress Tracking

---

## 🟢 PHASE 0 — Foundation Check
**Days 1–2** *(Quick setup & fundamentals check)*

### Day 1 — Python for Machine Learning
- Lists, Tuples, Dictionaries, Sets
- List & Dictionary Comprehensions
- Functions, *args, **kwargs, Type Hinting
- Lambda Functions & Functional Built-ins (`map`, `filter`)
- Exception Handling (`try`, `except`, `finally`, custom exceptions)
- OOP Basics (Classes, `__init__`, Methods, Inheritance)
- Modules, Packages & Virtual Environments (`venv`, `pip`)

### Day 2 — Data Science & Math Foundation
- **NumPy:** Arrays, `shape`, `reshape`, Indexing/Slicing, Broadcasting, Vectorized operations, Matrix multiplication (`@`).
- **Pandas:** Series & DataFrames, `loc` vs `iloc`, Boolean filtering, `groupby`, `merge`/`concat`, Missing value inspection (`isna`, `dropna`, `fillna`).
- **Math & Statistics:** Mean, Median, Mode, Variance, Standard Deviation, Probability basics, Vectors, Matrices, Dot product, Transpose.

---

## 🔵 PHASE 1 — Scikit-learn Core & Supervised Basics
**Days 3–16** *(Core estimators, classical algorithms & evaluation)*

### Day 3 — What is Scikit-learn & Estimator API
- What is Scikit-learn?
- Core ML Workflow (Data → Split → Preprocess → Fit → Predict → Evaluate)
- Features matrix ($X$) vs Target vector ($y$)
- Supervised vs Unsupervised Learning
- **Estimator Hierarchy:** Estimator (`fit`), Transformer (`transform`, `fit_transform`), Predictor (`predict`, `predict_proba`, `score`)
- Built-in Datasets exploration (`load_iris`, `load_wine`, `load_breast_cancer`)

### Day 4 — Train/Test Split & Data Leakage
- Why we split data (Generalization vs Memorization)
- `train_test_split()` parameters: `test_size`, `random_state`, `shuffle`, `stratify`
- **Data Leakage:** Definition, causes, consequences, and golden rules of train/test isolation

### Day 5 — Linear Regression Baseline
- Simple & Multiple Linear Regression (`LinearRegression`)
- Mathematical formulation: $y = \mathbf{w}^T \mathbf{x} + b$
- Ordinary Least Squares (OLS) closed-form solution vs Gradient Descent
- Inspecting learned parameters: `coef_`, `intercept_`
- Residuals and line of best fit

### Day 6 — Regression Evaluation Metrics
- Mean Absolute Error (`mean_absolute_error`)
- Mean Squared Error (`mean_squared_error`)
- Root Mean Squared Error (`root_mean_squared_error`)
- Coefficient of Determination ($R^2$ score: `r2_score`)
- Adjusted $R^2$ & choosing the right metric for outliers vs standard error

### Day 7 — Polynomial Regression & Overfitting
- Non-linear feature transformation with `PolynomialFeatures`
- Degree hyperparameter and polynomial interaction terms
- Underfitting (High Bias) vs Overfitting (High Variance)
- The Bias-Variance Trade-off

### Day 8 — Ridge Regression (L2 Regularization)
- Why OLS overfits with multicollinear features
- $L_2$ Regularization penalty ($\lambda \sum w_i^2$)
- `Ridge` estimator & `alpha` hyperparameter
- Weight shrinkage behavior & bias-variance impact

### Day 9 — Lasso Regression & ElasticNet
- $L_1$ Regularization penalty ($\lambda \sum |w_i|$)
- `Lasso` estimator: Sparsity & automatic feature selection
- Comparing $L_1$ vs $L_2$ geometry and behavior
- `ElasticNet` (combining $L_1 + L_2$ penalties with `l1_ratio`)

### Day 10 — Logistic Regression (Classification)
- Binary vs Multiclass Classification
- Sigmoid / Logistic Function: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- Odds, Log-Odds (Logits), and Probability output
- Decision Boundary & Decision Threshold
- `LogisticRegression` API: `predict()` vs `predict_proba()` vs `decision_function()`

### Day 11 — Classification Metrics
- Accuracy & the Accuracy Paradox in imbalanced datasets
- Confusion Matrix (`confusion_matrix`, `ConfusionMatrixDisplay`)
- Precision, Recall, Specificity, F1-Score (`f1_score`, `classification_report`)
- When Precision matters (Spam, Fraud) vs when Recall matters (Medical diagnosis, Safety)
- ROC Curve & ROC-AUC score (`roc_curve`, `roc_auc_score`)
- Precision-Recall Curve (`precision_recall_curve`, `average_precision_score`)

### Day 12 — k-Nearest Neighbors (k-NN)
- `KNeighborsClassifier` & `KNeighborsRegressor`
- Instance-based (Lazy) learning vs Parametric models
- Distance metrics (Euclidean $L_2$, Manhattan $L_1$, Minkowski $L_p$)
- Hyperparameter $k$ (`n_neighbors`) & Bias-Variance tradeoff
- Distance weighting (`weights='distance'`)
- Why Feature Scaling is mandatory for distance-based models

### Day 13 — Naive Bayes Classifier
- Probability Foundation: Prior $P(y)$, Likelihood $P(X|y)$, Marginal $P(X)$, Posterior $P(y|X)$
- Bayes' Theorem & Conditional Independence ("Naive") assumption
- Log-probabilities to prevent numerical underflow
- **Variants in Scikit-Learn:**
  - `GaussianNB`: Continuous normally distributed features
  - `MultinomialNB`: Discrete counts / frequency data (Bag-of-Words)
  - `BernoulliNB`: Binary indicators (0 or 1 word presence)
  - `ComplementNB`: Imbalanced multiclass text classification
- Zero-frequency problem & Laplace / Additive Smoothing (`alpha` parameter)
- Feature extraction basics (`CountVectorizer`, `TfidfVectorizer`)

### Day 14 — Support Vector Machines (SVM)
- Maximum Margin Classifier concept & Support Vectors
- `SVC` (Classification) & `SVR` (Regression)
- Hard margin vs Soft margin (Slack variables)
- Regularization parameter `C` (penalty for misclassifications)
- Kernel Trick: Non-linear projection to high-dimensional space
- Kernel functions: `linear`, `poly` (degree), `rbf` (Gaussian Radial Basis Function), `sigmoid`
- Kernel coefficient `gamma` (radius of influence for support vectors)
- Why feature scaling is mandatory for SVM

### Day 15 — Decision Trees
- Tree architecture: Root node, Internal decision nodes, Leaf/terminal nodes
- `DecisionTreeClassifier` & `DecisionTreeRegressor`
- Split criteria: Gini Impurity vs Information Gain / Entropy
- Regression split criterion: Mean Squared Error (variance reduction)
- Tree depth & regularization hyperparameters: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- Feature importance calculation (Mean Decrease in Impurity)
- Visualizing decision trees with `plot_tree()` and `export_text()`

### Day 16 — Baseline ML Benchmark Project
- Build a standardized evaluation harness
- Compare 5 core classifiers on a real-world dataset (e.g. Wine / Breast Cancer / Titanic):
  1. Logistic Regression
  2. k-Nearest Neighbors
  3. Naive Bayes
  4. Support Vector Machine (SVC)
  5. Decision Tree
- Generate comparative metrics summary (Accuracy, Precision, Recall, F1, ROC-AUC, Training Time)

---

## 🟡 PHASE 2 — Preprocessing, Pipelines & Ensembles
**Days 17–30** *(Production preprocessing, leak-proof pipelines & ensemble learning)*

### Day 17 — Feature Scaling
- `StandardScaler` (Z-score standardization: mean=0, std=1)
- `MinMaxScaler` (Bounded scaling to [0, 1] range)
- `RobustScaler` (Median & IQR scaling for outlier resilience)
- `MaxAbsScaler` (Sparse matrix scaling)
- Standardization vs Normalization: When to use which
- Golden Rule: Fit scaler ONLY on training data, transform test data

### Day 18 — Categorical Encoding
- Nominal vs Ordinal categorical features
- `OneHotEncoder` (Dummy encoding, `drop='first'`, `sparse_output=False`)
- Handling unseen categories in production: `handle_unknown='ignore'`
- `OrdinalEncoder` for ranked/ordered categories (`categories=[...]`)

### Day 19 — Missing Value Imputation
- Missing Data Mechanisms: MCAR (Missing Completely at Random), MAR (Missing at Random), MNAR (Missing Not at Random)
- `SimpleImputer` strategies (`mean`, `median`, `most_frequent`, `constant`)
- `KNNImputer` (Distance-weighted nearest neighbor imputation)
- Missing value indicator tracking (`add_indicator=True`)

### Day 20 — ColumnTransformer (Heterogeneous Data Preprocessing)
- `ColumnTransformer` & `make_column_transformer`
- Applying distinct transformations to numeric vs categorical columns in a single step
- `remainder='passthrough'` vs `remainder='drop'`
- Extracting output feature names: `get_feature_names_out()`

### Day 21 — Scikit-learn Pipelines
- `Pipeline` & `make_pipeline`
- Chaining Imputer $\to$ Scaler/Encoder $\to$ Model into an atomic estimator
- Complete prevention of data leakage during evaluation and cross-validation
- Accessing named steps: `pipeline.named_steps`
- Calling `fit()` and `predict()` on composite workflows

### Day 22 — Cross-Validation
- Why single train/test split can be misleading and high variance
- `KFold` cross-validation
- `StratifiedKFold` for classification (preserving class balance in folds)
- Evaluation helpers: `cross_val_score()` & `cross_validate()` (multi-metric evaluation)
- Out-of-fold predictions with `cross_val_predict()`

### Day 23 — Hyperparameter Tuning: GridSearchCV
- Hyperparameters vs Learned Model Parameters
- Exhaustive search with `GridSearchCV`
- Defining parameter grids (including pipeline step parameters: `step__param`)
- Inspecting results: `best_params_`, `best_score_`, `best_estimator_`, `cv_results_`

### Day 24 — Fast Tuning: RandomizedSearchCV & HalvingGridSearchCV
- Curse of dimensionality in exhaustive grid search
- Random sampling over parameter distributions with `RandomizedSearchCV`
- `n_iter` parameter and statistical efficiency
- Successive Halving for large parameter spaces (`HalvingGridSearchCV`, `HalvingRandomSearchCV`)

### Day 25 — Bagging & Random Forest
- Bootstrap Aggregation (Bagging) concept
- `RandomForestClassifier` & `RandomForestRegressor`
- Key hyperparameters: `n_estimators`, `max_depth`, `max_features`, `min_samples_split`
- Out-of-Bag error estimation (`oob_score=True`)
- Ensemble feature importances

### Day 26 — Extremely Randomized Trees (Extra Trees)
- `ExtraTreesClassifier` & `ExtraTreesRegressor`
- Random threshold splits vs optimal splits
- Random Forest vs Extra Trees (Computational speed vs Variance reduction)

### Day 27 — Boosting Fundamentals: AdaBoost
- Boosting concept: Sequential ensemble learning from mistakes
- Sample weighting and focus on hard-to-classify examples
- `AdaBoostClassifier` & `AdaBoostRegressor`
- Base estimators (Decision Stumps), `learning_rate`, `n_estimators`

### Day 28 — Gradient Boosting (GBDT)
- Gradient Boosting intuition: Fitting trees to pseudo-residuals / negative gradients
- `GradientBoostingClassifier` & `GradientBoostingRegressor`
- Loss functions (deviance, exponential, squared_error, absolute_error, huber)
- Shrinkage (`learning_rate`), Subsampling (`subsample` / Stochastic Gradient Boosting)

### Day 29 — HistGradientBoosting (High-Speed Boosting)
- Histogram-based binning algorithm (Scikit-learn's native LightGBM-style model)
- `HistGradientBoostingClassifier` & `HistGradientBoostingRegressor`
- Native support for missing values (no imputer required!)
- Native support for categorical features (`categorical_features`)
- Extreme performance on large datasets ($N > 10,000$)

### Day 30 — Voting & Stacking Ensembles
- **Voting Ensembles:** `VotingClassifier` & `VotingRegressor` (Hard voting vs Soft probability-weighted voting)
- **Stacking Ensembles:** `StackingClassifier` & `StackingRegressor`
- Multi-layer architecture: Base models (Layer 1) $\to$ Meta-Learner (Layer 2)
- Internal cross-validation in stacking to prevent meta-learner data leakage

---

## 🟠 PHASE 3 — Unsupervised Learning, Feature Selection & Explainability
**Days 31–45** *(Clustering, dimensionality reduction, feature selection, text & interpretability)*

### Day 31 — K-Means Clustering
- Unsupervised learning principles (No target labels $y$)
- `KMeans` & `MiniBatchKMeans` algorithm
- Centroids, Voronoi partitions, Within-Cluster Sum of Squares (Inertia)
- `fit()`, `predict()`, `transform()` (distance to cluster centers)
- Initialization with `k-means++`

### Day 32 — Density-Based Clustering (DBSCAN)
- `DBSCAN` (Density-Based Spatial Clustering of Applications with Noise)
- Core points, Border points, Noise points
- Hyperparameters: `eps` (neighborhood radius), `min_samples`
- Discovering arbitrarily shaped clusters without specifying cluster count $k$

### Day 33 — Hierarchical / Agglomerative Clustering
- `AgglomerativeClustering` (Bottom-up hierarchical merging)
- Linkage criteria: `ward`, `complete`, `average`, `single`
- Affinity / Distance metrics
- Dendrogram visualization concept using SciPy (`scipy.cluster.hierarchy`)

### Day 34 — Cluster Evaluation & Optimal K
- The Elbow Method (Inertia vs Number of Clusters)
- Silhouette Coefficient (`silhouette_score`, `silhouette_samples`)
- Davies-Bouldin Index (`davies_bouldin_score`)
- Selecting the optimal number of clusters

### Day 35 — Principal Component Analysis (PCA)
- `PCA` (Orthogonal linear transformation for dimensionality reduction)
- Eigenvalues, Eigenvectors, Covariance Matrix
- Principal components & Explained Variance Ratio (`explained_variance_ratio_`)
- Cumulative explained variance & selecting the number of components
- 2D/3D Data visualization of high-dimensional datasets

### Day 36 — TruncatedSVD & Manifold Learning
- Singular Value Decomposition on Sparse Matrices (`TruncatedSVD`)
- Differences between PCA (dense, centered) vs TruncatedSVD (sparse, uncentered)
- Latent Semantic Analysis (LSA) for text documents
- Manifold learning intuition (t-SNE / UMAP for non-linear visualization)

### Day 37 — Filter-Based Feature Selection
- Statistical feature filtering methods
- `VarianceThreshold` (Removing zero-variance and quasi-constant features)
- Univariate Statistical Tests: `SelectKBest`, `SelectPercentile`
- Scoring functions: `f_classif`, `f_regression`, `chi2`, `mutual_info_classif`, `mutual_info_regression`

### Day 38 — Wrapper & Model-Based Feature Selection
- Recursive Feature Elimination: `RFE` & `RFECV` (RFE with Cross-Validation)
- Model-Based Selection: `SelectFromModel` (using tree importances or Lasso $L_1$ coefficients)
- Comparing Filter vs Wrapper vs Embedded feature selection methods

### Day 39 — Text Feature Extraction: CountVectorizer
- Natural Language Processing (NLP) text representation
- `CountVectorizer` (Bag-of-Words model)
- Tokenization, n-grams (`ngram_range`), vocabulary dictionary
- Stop words removal, min/max document frequency (`min_df`, `max_df`)

### Day 40 — Text Feature Extraction: TF-IDF
- `TfidfTransformer` & `TfidfVectorizer`
- Term Frequency ($TF$) $\times$ Inverse Document Frequency ($IDF$)
- Sublinear TF scaling, smooth IDF, $L_2$ document normalization
- When to use CountVectorizer vs TfidfVectorizer

### Day 41 — End-to-End NLP Text Classification Project
- Build a real-world text classification system (e.g. Spam / Sentiment / News categorization)
- Construct pipeline: `Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])`
- Hyperparameter tuning of n-gram ranges, sublinear TF, and classifier smoothing
- Out-of-sample inference on raw custom text strings

### Day 42 — Handling Imbalanced Datasets
- Imbalanced classification challenges & failure of raw accuracy
- Algorithmic balancing: `class_weight='balanced'` in Linear, Tree, and SVM models
- Precision-Recall curves & PR-AUC (`PrecisionRecallDisplay`)
- Probability decision threshold tuning (`predict_proba >= custom_threshold`)
- Resampling overview (Oversampling / SMOTE vs Undersampling concepts)

### Day 43 — Probability Calibration
- Why some models output uncalibrated probabilities (SVM, Naive Bayes, Boosted Trees)
- `CalibratedClassifierCV`
- Calibration methods: Sigmoid / Platt Scaling (`method='sigmoid'`) vs Non-parametric Isotonic Regression (`method='isotonic'`)
- Reliability Curves / Calibration Displays (`CalibrationDisplay`)
- Brier Score loss (`brier_score_loss`)

### Day 44 — Permutation Feature Importance & Tree Importances
- Flaws of default tree MDI (Mean Decrease in Impurity) feature importances (bias toward high-cardinality features)
- `permutation_importance` (Model-agnostic evaluation on test data)
- How permutation importance works (measuring metric drops upon shuffling features)
- Comparing training vs test permutation importances

### Day 45 — Partial Dependence & Model Explainability
- `PartialDependenceDisplay` (`from_estimator`)
- Partial Dependence Plots (PDP): Marginal effect of 1 or 2 features on model predictions
- Individual Conditional Expectation (ICE) plots: Uncovering feature interactions and sub-population effects
- Interpreting black-box models before production

---

## 🔴 PHASE 4 — Advanced Scikit-learn + ML Engineering
**Days 46–60** *(Custom components, out-of-core scaling, serialization, MLOps, APIs & Docker deployment)*

### Day 46 — Custom Transformers
- Scikit-Learn Base Architecture: `BaseEstimator`, `TransformerMixin`
- Building Scikit-Learn compliant custom transformers
- Implementing `fit()`, `transform()`, and `get_feature_names_out()`
- Custom feature engineering, log-transforms, domain-specific feature extractors
- Integrating custom transformers directly inside `Pipeline` and `ColumnTransformer`

### Day 47 — Custom Estimators
- Building Scikit-Learn compliant custom models
- `ClassifierMixin` & `RegressorMixin`
- Implementing `fit()`, `predict()`, `predict_proba()`, `score()`
- Automatic compatibility with `cross_val_score`, `GridSearchCV`, and `Pipeline`

### Day 48 — Custom Metrics & Scorers
- Writing custom business-specific loss & evaluation functions
- `make_scorer()` utility
- Defining `greater_is_better` and `response_method` (`predict` vs `predict_proba`)
- Passing custom scoring functions to `cross_val_score` and `GridSearchCV`

### Day 49 — Stochastic Gradient Descent & Incremental Learning
- Online / Out-of-Core learning for datasets larger than RAM
- `SGDClassifier` & `SGDRegressor`
- Incremental mini-batch training using `partial_fit()`
- Streaming data processing with `HashingVectorizer` and generators

### Day 50 — Model Serialization & Persistence
- Model saving & loading with `joblib` (`joblib.dump`, `joblib.load`)
- Saving full end-to-end pipelines (Preprocessor + Model bundled together)
- Serialization best practices, version pinning, security risks with pickle

### Day 51 — Model Validation & Learning Curves
- `learning_curve`: Training score vs Validation score across growing sample sizes
- `validation_curve`: Model performance across single hyperparameter variations
- Diagnosing High Bias (Underfitting) vs High Variance (Overfitting) quantitatively
- Determining if collecting more training data will improve model accuracy

### Day 52 — Model Interpretation with SHAP
- Introduction to SHAP (SHapley Additive exPlanations)
- Game-theoretic feature attributions
- `TreeExplainer` for tree models & `KernelExplainer` for arbitrary Scikit-Learn pipelines
- Summary plots, Waterfall plots, and Force plots (Global vs Local explanations)

### Day 53 — Modern Hyperparameter Optimization with Optuna
- Bayesian Optimization vs Random/Grid Search
- Integrating Scikit-Learn with Optuna
- Defining search spaces, objective functions, trials
- Optuna pruning callbacks for fast early-stopping of unpromising trials

### Day 54 — ONNX Model Export & High-Performance Inference
- What is ONNX (Open Neural Network Exchange)?
- Converting Scikit-Learn pipelines to ONNX format using `skl2onnx`
- Running ultra-low latency predictions with `onnxruntime`
- Cross-platform portability (running Scikit-Learn models in C++, Go, Java, or Edge devices)

### Day 55 — Serving Models with FastAPI
- Wrapping trained Scikit-Learn pipelines in a production REST API
- Building a FastAPI application with Pydantic request/response validation schemas
- `/health` and `/predict` endpoints
- Error handling, batch inference endpoints, JSON input/output formatting

### Day 56 — Docker Containerization for ML APIs
- Docker fundamentals for Machine Learning
- Writing an optimized production `Dockerfile` for Python + Scikit-Learn + FastAPI
- `.dockerignore`, dependency locking (`requirements.txt`), non-root container user
- Building and running the ML Docker container locally: `docker build` & `docker run`

### Day 57 — Experiment Tracking with MLflow
- Introduction to MLflow Tracking
- Logging hyperparameters (`mlflow.log_params`), evaluation metrics (`mlflow.log_metrics`), and artifacts
- Logging Scikit-Learn pipeline models with `mlflow.sklearn.log_model`
- Visualizing runs and comparing models in the MLflow UI

### Day 58 — Production Model Monitoring & Drift Detection
- Data Drift (Covariate Shift: $P(X)$ changes)
- Concept Drift (Relationship shift: $P(y|X)$ changes)
- Detecting feature distribution drift using Kolmogorov-Smirnov (KS) test & Population Stability Index (PSI)
- Setting up alerts and automated retraining triggers

### Day 59 — Final Capstone Project: Part 1 — End-to-End Pipeline, Tuning & Artifacts
- **Problem Statement:** Real-world enterprise dataset (e.g. Customer Churn / Credit Default / Housing Price)
- **Pipeline Architecture:**
  1. Exploratory Data Analysis & Schema definition
  2. Custom Transformers for domain feature engineering
  3. `ColumnTransformer` (Imputation, Scaling, One-Hot Encoding)
  4. Candidate Model Benchmarking (HistGradientBoosting, Random Forest, SVC)
  5. Bayesian Hyperparameter Optimization with Optuna
  6. Experiment logging with MLflow
  7. Exporting production-ready serialized pipeline (`joblib` and `ONNX`)

### Day 60 — Final Capstone Project: Part 2 — Full-Stack Deployment & UI
- **Deployment Architecture:**
  1. FastAPI application serving the serialized Capstone pipeline
  2. Containerizing the complete application with Docker
  3. Interactive Web UI / Frontend demo for real-time model inference
  4. End-to-end verification, automated tests, and GitHub repository finalization!
