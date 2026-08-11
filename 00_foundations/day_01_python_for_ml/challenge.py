# """
# DAY 01 ACTIVE RECALL CHALLENGE: Python for Machine Learning

# TASK 1: Data Structures
# 1. Create a list of 3 feature names: ["age", "salary", "experience"]
features = ["age", "salary", "experience"]
print(features)

# 2. Create a tuple representing dataset shape: 500 rows, 5 columns
dataset_shape = (500, 5)
rows, columns = dataset_shape
print(f"Rows: {rows}, Columns: {columns}")

# 3. Create a dictionary with model hyperparams:
# name="DecisionTree", max_depth=3
model_params = {"name": "DecisionTree", "max_depth": 3}
print(model_params)

# TASK 2: List Comprehension
# Given a list of prices: [100, 250, 400, 50], create a new list containing
# prices greater than 150 with a 10% discount applied (price * 0.9).
prices = [100, 250, 400, 50]
discounted_prices = [price * 0.9 for price in prices if price > 150]
print(discounted_prices)

# TASK 3: Lambda & Functions
# Write a lambda function `calc_loss` that takes (y_true, y_pred)
# and returns the absolute difference: abs(y_true - y_pred).
calc_loss = lambda y_true, y_pred: abs(y_true - y_pred)
print(calc_loss(100, 85))

# TASK 4: Custom Class (Scikit-Learn Pattern)
# Create a class `SimpleScaler` with:
# - __init__(self)
# - fit(self, X): computes self.min_val_ and self.max_val_ from list X
# - transform(self, X): returns
#   [(x - self.min_val_) / (self.max_val_ - self.min_val_) for x in X]
class SimpleScaler:

    def __init__(self):
        self.min_val_ = None
        self.max_val_ = None

    def fit(self, X):
        self.min_val_ = min(X)
        self.max_val_ = max(X)

    def transform(self, X):
        return [(x - self.min_val_) / (self.max_val_ - self.min_val_) for x in X]


X = [10, 20, 30, 40, 50]

scaler = SimpleScaler()

scaler.fit(X)

result = scaler.transform(X)

print(result)
