"""
===================================================================
DAY 01: Python for Machine Learning (Core Fundamentals)
===================================================================
Topics Covered:
1. Data Structures (Lists, Tuples, Dictionaries, Sets)
2. List & Dictionary Comprehensions
3. Functions, *args, **kwargs, & Lambda Expressions
4. Exception Handling
5. Object-Oriented Programming (OOP) for Scikit-Learn
"""

# -----------------------------------------------------------------
# 1. DATA STRUCTURES
# -----------------------------------------------------------------

# List (Mutable, ordered)
features = ["age", "income", "credit_score"]
features.append("debt_ratio")
print("Features List:", features)

# Tuple (Immutable, ordered - used for dataset shapes e.g., (100, 4))
dataset_shape = (1000, 10)
print(f"Rows: {dataset_shape[0]}, Columns: {dataset_shape[1]}")

# Dictionary (Key-Value pairs - used for model hyperparameters & metrics)
model_config = {
    "model_name": "RandomForest",
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42
}
print("Model Config:", model_config["model_name"])
print("Safe Get max_depth:", model_config.get("max_depth", 10))

# Set (Unique elements, unordered - used for feature deduplication)
raw_labels = ["cat", "dog", "cat", "bird", "dog"]
unique_classes = set(raw_labels)
print("Unique Classes:", unique_classes)


# -----------------------------------------------------------------
# 2. COMPREHENSIONS (Fast Data Transformation)
# -----------------------------------------------------------------

# Squaring numbers (List Comprehension)
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers if x % 2 == 0]
print("Squared Even Numbers:", squared)

# Normalizing feature names (Dictionary Comprehension)
column_names = [" Age ", " Annual Income ", "Credit Score"]
clean_columns = {col: col.strip().lower().replace(" ", "_") for col in column_names}
print("Cleaned Columns Mapping:", clean_columns)


# -----------------------------------------------------------------
# 3. FUNCTIONS, LAMBDA, & *args / **kwargs
# -----------------------------------------------------------------

def evaluate_model(model_name, accuracy, **metrics):
    """Flexible function using **kwargs for custom metrics."""
    print(f"\n--- Evaluation for {model_name} ---")
    print(f"Accuracy: {accuracy:.2%}")
    for metric_name, value in metrics.items():
        print(f"{metric_name.capitalize()}: {value:.4f}")

evaluate_model("LogisticRegression", 0.9234, precision=0.912, recall=0.945, f1=0.928)

# Lambda function (Anonymous short function - used in Pandas apply)
normalize = lambda x, min_val, max_val: (x - min_val) / (max_val - min_val)
print("Scaled value (50 in range 0-100):", normalize(50, 0, 100))


# -----------------------------------------------------------------
# 4. EXCEPTION HANDLING (Robust ML Code)
# -----------------------------------------------------------------

def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Warning: Division by zero encountered! Returning 0.0")
        result = 0.0
    except TypeError as e:
        print(f"Type Error: {e}")
        result = None
    finally:
        print("Division operation attempted.")
    return result

print("Safe Division Result:", safe_divide(10, 0))


# -----------------------------------------------------------------
# 5. OBJECT-ORIENTED PROGRAMMING (OOP) - ESTIMATOR PATTERN
# -----------------------------------------------------------------

class DummyModel:
    """A minimal custom estimator mimicking Scikit-Learn's API."""
    def __init__(self, multiplier=2.0):
        self.multiplier = multiplier
        self.is_fitted_ = False
        
    def fit(self, X, y):
        """Fit phase: learn parameters from training data."""
        self.mean_x_ = sum(X) / len(X)
        self.is_fitted_ = True
        print(f"[DummyModel] Fitted with mean_x = {self.mean_x_:.2f}")
        return self
        
    def predict(self, X):
        """Predict phase: apply learned rule to new data."""
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet! Call fit() first.")
        return [x * self.multiplier for x in X]

# Test DummyModel
X_dummy = [10, 20, 30, 40]
y_dummy = [20, 40, 60, 80]

model = DummyModel(multiplier=2.0)
model.fit(X_dummy, y_dummy)
predictions = model.predict([5, 15, 25])
print("Dummy Model Predictions:", predictions)
