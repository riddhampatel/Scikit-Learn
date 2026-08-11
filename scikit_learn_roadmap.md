🧠 Complete Scikit-learn Roadmap
Beginner → Advanced → Production

Recommended duration: ~60 days

---

🟢 PHASE 0 — Foundation Check
Days 1–2

You already know most of this, so don't spend weeks here.

Day 1 — Python for ML
- Lists
- Tuples
- Dictionaries
- Sets
- List comprehensions
- Functions
- Lambda
- Exception handling
- OOP basics
- Modules/packages
- Virtual environments

Day 2 — Data Science Foundation

NumPy
- Arrays
- Shape
- Reshape
- Indexing
- Broadcasting
- Vector operations

Pandas
- DataFrame
- Series
- loc & iloc
- groupby
- merge
- Missing values (dropna, fillna)

Math
- Mean, Median, Variance, Standard deviation
- Probability
- Vectors, Matrices, Dot product

---

🔵 PHASE 1 — Scikit-learn Core & Supervised Basics
Days 3–15

This is where your real Scikit-learn journey starts.

Day 3 — What is Scikit-learn?

Learn:
- What is Scikit-learn?
- ML workflow
- Dataset, Features, Target (X, y)
- Supervised vs Unsupervised learning
- Estimator, Transformer, Predictor

Understand the basic API:
- fit()
- predict()
- transform()
- fit_transform()
- score()

Practice:
- Load Iris: from sklearn.datasets import load_iris
- Explore: X, y, X.shape, y.shape

Day 4 — Train/Test Split

Learn:
- train_test_split()

Understand:
- Training set vs Testing set
- test_size
- random_state
- shuffle
- Data leakage & why we split data

Practice:
- Split a dataset into 80% Training and 20% Testing.

Day 5 — First Regression

Learn:
- Linear Regression (LinearRegression)

Understand:
- Independent vs Dependent variables
- Coefficient
- Intercept
- Prediction & Residuals
- Line of best fit

Practice:
- Build a simple House Price Predictor.

Day 6 — Regression Evaluation

Learn:
- MAE: mean_absolute_error()
- MSE: mean_squared_error()
- RMSE: sqrt(MSE)
- R² Score: r2_score()

Understand:
- Which metric to use and why.

Day 7 — Polynomial Regression

Learn:
- PolynomialFeatures

Understand:
- Linear vs Polynomial relationship
- Degree hyperparameter
- Underfitting vs Overfitting

Practice:
- Fit non-linear data using PolynomialFeatures.

Day 8 — Ridge Regression

Learn:
- Ridge

Understand:
- Regularization
- L2 penalty
- Overfitting control
- alpha parameter

Day 9 — Lasso Regression & ElasticNet

Learn:
- Lasso & ElasticNet

Understand:
- L1 penalty & automatic feature selection
- Comparing L1 vs L2 regularization
- ElasticNet (L1 + L2 combined)

Day 10 — Logistic Regression (Classification)

Learn:
- LogisticRegression

Understand:
- Binary classification
- Sigmoid function & probabilities
- Decision boundaries
- predict() vs predict_proba()

Practice:
- Classify breast cancer tumor samples.

Day 11 — Classification Metrics

Learn:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (ConfusionMatrixDisplay)
- classification_report()
- ROC Curve & ROC-AUC score

Understand:
- When Precision matters vs when Recall matters.

Day 12 — k-Nearest Neighbors (k-NN)

Learn:
- KNeighborsClassifier & KNeighborsRegressor

Understand:
- Distance metrics (Euclidean, Manhattan)
- Impact of k parameter
- Feature scaling requirement

Day 13 — Naive Bayes Classifier

Learn:
- GaussianNB, MultinomialNB

Understand:
- Bayes Theorem
- Independence assumption
- Text classification suitability

Day 14 — Decision Trees

Learn:
- DecisionTreeClassifier & DecisionTreeRegressor

Understand:
- Gini impurity vs Entropy
- Tree depth & max_depth
- Feature importance in trees

Day 15 — Baseline Project: Classification & Regression

Practice:
- Build and compare 3 baseline models on a dataset.
- Generate a summary report of test scores.

---

🟡 PHASE 2 — Preprocessing, Pipelines & Ensembles
Days 16–30

Day 16 — Data Scaling

Learn:
- StandardScaler
- MinMaxScaler
- RobustScaler

Understand:
- Why scale features?
- Fit scaler ONLY on train data, transform test data!

Day 17 — Categorical Encoding

Learn:
- OneHotEncoder
- OrdinalEncoder

Understand:
- Nominal vs Ordinal data
- Dummy variable trap & handle_unknown='ignore'

Day 18 — Missing Value Imputation

Learn:
- SimpleImputer (mean, median, most_frequent, constant)
- KNNImputer

Understand:
- Missing Data mechanisms (MCAR, MAR, MNAR)

Day 19 — ColumnTransformer

Learn:
- ColumnTransformer

Understand:
- Applying different transformers to numeric vs categorical columns simultaneously.

Day 20 — Scikit-learn Pipelines

Learn:
- Pipeline

Understand:
- Chaining Preprocessing + Model together
- Preventing data leakage completely
- pipeline.fit() and pipeline.predict()

Practice:
- Build a single Pipeline that imputes, encodes, scales, and fits a model.

Day 21 — Cross-Validation

Learn:
- KFold
- StratifiedKFold
- cross_val_score & cross_validate

Understand:
- Why train/test split isn't enough
- Stratification for imbalanced datasets

Day 22 — Grid Search CV

Learn:
- GridSearchCV

Understand:
- Hyperparameter tuning
- Search grid definition
- best_params_, best_score_, best_estimator_

Day 23 — RandomizedSearchCV & HalvingGridSearchCV

Learn:
- RandomizedSearchCV
- HalvingGridSearchCV

Understand:
- Random sampling vs Grid search
- Successive halving for fast search on large parameter spaces

Day 24 — Bagging & Random Forest

Learn:
- RandomForestClassifier & RandomForestRegressor

Understand:
- Bootstrap aggregation (Bagging)
- n_estimators, max_features, oob_score
- Why Random Forest rarely overfits easily

Day 25 — Extra Trees

Learn:
- ExtraTreesClassifier & ExtraTreesRegressor

Understand:
- Random splits vs optimal splits
- Speed vs variance reduction

Day 26 — Boosting Basics (AdaBoost)

Learn:
- AdaBoostClassifier & AdaBoostRegressor

Understand:
- Sequential learning
- Weighted sample learning from errors

Day 27 — Gradient Boosting

Learn:
- GradientBoostingClassifier & GradientBoostingRegressor

Understand:
- Residual learning
- learning_rate, n_estimators, subsample

Day 28 — HistGradientBoosting (Fast Boosting)

Learn:
- HistGradientBoostingClassifier & HistGradientBoostingRegressor

Understand:
- Histogram-based binning (Scikit-learn's native LightGBM algorithm)
- Native missing value support (no imputer needed!)
- Extreme speed on large datasets (> 10,000 samples)

Day 29 — Voting & Stacking Ensembles

Learn:
- VotingClassifier & VotingRegressor
- StackingClassifier & StackingRegressor

Understand:
- Hard vs Soft voting
- Meta-learner in Stacking

Day 30 — Capstone Project: End-to-End Pipeline & Ensembles

Practice:
- Build a full ColumnTransformer + Pipeline + HistGradientBoosting + GridSearchCV model on Titanic or Housing dataset.

---

🟠 PHASE 3 — Unsupervised Learning, Feature Selection & Explainability
Days 31–45

Day 31 — K-Means Clustering

Learn:
- KMeans

Understand:
- Centroids, inertia
- fit(), predict(), transform()

Day 32 — DBSCAN & Agglomerative Clustering

Learn:
- DBSCAN, AgglomerativeClustering

Understand:
- Density-based clustering vs distance-based
- eps, min_samples
- Hierarchical dendrograms

Day 33 — Cluster Evaluation

Learn:
- Elbow method
- Silhouette score (silhouette_score)

Understand:
- Finding optimal k in clustering

Day 34 — Principal Component Analysis (PCA)

Learn:
- PCA

Understand:
- Variance explained (explained_variance_ratio_)
- Reducing dimensions from N features to K components

Day 35 — TruncatedSVD & Manifold Learning

Learn:
- TruncatedSVD

Understand:
- Sparse matrix dimensionality reduction (great for text!)

Day 36 — Filter Feature Selection

Learn:
- SelectKBest, SelectPercentile
- f_classif, f_regression, mutual_info_classif

Understand:
- Selecting top features using statistical tests

Day 37 — Wrapper & Model Feature Selection

Learn:
- RFE (Recursive Feature Elimination)
- SelectFromModel

Understand:
- Iterative feature pruning using model feature importances

Day 38 — Text Feature Extraction: CountVectorizer

Learn:
- CountVectorizer

Understand:
- Bag of Words representation
- stop_words, ngram_range, max_features

Day 39 — Text Feature Extraction: TF-IDF

Learn:
- TfidfVectorizer

Understand:
- Term Frequency - Inverse Document Frequency
- Penalizing frequent words across documents

Day 40 — NLP Spam Classification Project

Practice:
- Build an SMS/Email Spam Classifier using TfidfVectorizer + MultinomialNB in a Pipeline.

Day 41 — Class Imbalance Handling

Learn:
- class_weight='balanced'
- Precision-Recall Curve & PR-AUC

Understand:
- Why accuracy lies on imbalanced datasets
- Threshold tuning with predict_proba()

Day 42 — Probability Calibration

Learn:
- CalibratedClassifierCV

Understand:
- Sigmoid vs Isotonic calibration
- Making predict_proba outputs reflect true empirical probabilities

Day 43 — Permutation Importance

Learn:
- permutation_importance

Understand:
- Model-agnostic feature importance
- Why default tree feature importances can be misleading

Day 44 — Partial Dependence Plots

Learn:
- PartialDependenceDisplay

Understand:
- Visualizing how individual features influence model predictions

Day 45 — Intermediate Capstone: Customer Churn & Model Inspection

Practice:
- Build a Churn Prediction pipeline with class weighting, evaluate with PR-AUC, and extract Permutation Importances.

---

🔴 PHASE 4 — Advanced Customization, Serialization & MLOps
Days 46–60

Day 46 — Custom Transformers

Learn:
- BaseEstimator & TransformerMixin

Understand:
- Writing custom fit() and transform() methods
- Building domain-specific feature engineering modules for Pipelines

Day 47 — Custom Estimators

Learn:
- BaseEstimator & ClassifierMixin / RegressorMixin

Understand:
- Building custom Scikit-learn compatible classifiers from scratch

Day 48 — Custom Metrics & Scorers

Learn:
- make_scorer

Understand:
- Converting custom business loss functions into Scikit-learn scoring parameters for GridSearchCV

Day 49 — Out-of-Core Learning

Learn:
- partial_fit()
- SGDClassifier & SGDRegressor

Understand:
- Training models on datasets larger than RAM in streaming batches

Day 50 — Model Serialization

Learn:
- joblib.dump() & joblib.load()

Understand:
- Saving trained pipelines to disk and loading them for inference

Day 51 — ONNX Model Export

Learn:
- skl2onnx

Understand:
- Converting Scikit-learn models to ONNX format for high-speed inference in C++, Rust, Go, or ONNX Runtime

Day 52 — Advanced Ecosystem: category_encoders

Learn:
- TargetEncoder, CatBoostEncoder

Understand:
- Encoding high-cardinality categorical features without memory explosion

Day 53 — Advanced Ecosystem: Optuna Tuning

Learn:
- Optuna integration with Scikit-learn

Understand:
- Bayesian hyperparameter optimization faster than GridSearchCV

Day 54 — Model Explainability with SHAP

Learn:
- shap.Explainer

Understand:
- SHAP values for global and local prediction explanations

Day 55 — Serving Models with FastAPI

Learn:
- FastAPI + joblib

Understand:
- Creating a POST endpoint (/predict) that receives JSON inputs and returns model predictions

Day 56 — Dockerizing Scikit-learn Applications

Learn:
- Dockerfile writing for Python & Scikit-learn APIs

Understand:
- Containerizing your model REST API for cloud deployment

Day 57 — Experiment Tracking with MLflow

Learn:
- mlflow.sklearn.log_model()

Understand:
- Logging parameters, metrics, and pipeline artifacts automatically

Day 58 — Model Monitoring & Drift Detection

Learn:
- Data drift vs Concept drift basics

Understand:
- Monitoring feature distributions in production vs training data

Day 59 — MLOps Capstone Project: Part 1

Practice:
- Build a custom transformer pipeline, tune with Optuna, log with MLflow, and export with joblib/ONNX.

Day 60 — MLOps Capstone Project: Part 2

Practice:
- Wrap the trained model in a FastAPI endpoint, write a Dockerfile, build the image, and run predictions inside Docker.
