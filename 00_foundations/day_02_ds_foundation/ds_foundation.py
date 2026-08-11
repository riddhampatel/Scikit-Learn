"""
===================================================================
DAY 02: Data Science Foundation (NumPy, Pandas & ML Math)
===================================================================
Topics Covered:
1. NumPy: Vectors, Matrices, Reshaping, & Dot Product
2. Pandas: DataFrames, loc vs iloc, GroupBy, & Missing Values
3. ML Math Essentials: Mean, Variance, Standard Deviation, & Scaling
"""

import numpy as np
import pandas as pd

# -----------------------------------------------------------------
# 1. NUMPY CORE (High-Performance Vector Computation)
# -----------------------------------------------------------------
print("=== 1. NUMPY BASICS ===")

# Creating 1D Array (Vector) and 2D Array (Matrix)
arr_1d = np.array([10, 20, 30, 40, 50])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

print("1D Array Shape:", arr_1d.shape)
print("2D Array Shape:", arr_2d.shape)

# Reshaping 1D -> 2D Matrix (Crucial for Scikit-Learn X feature matrices!)
X_feature = arr_1d.reshape(-1, 1)  # Reshape to (5, 1) column vector
print("Reshaped X_feature Shape:", X_feature.shape)

# Vectorized Operations (No python loops needed!)
scaled_arr = arr_1d * 2.5
print("Vectorized Multiply:", scaled_arr)

# Matrix Multiplication / Dot Product (used in Linear Regression: y = X * w)
weights = np.array([0.5, 1.5])
inputs = np.array([[2, 4], [1, 3]])
predictions = np.dot(inputs, weights)
print("Linear Algebra Dot Product Predictions:", predictions)


# -----------------------------------------------------------------
# 2. PANDAS CORE (Data Manipulation & Cleaning)
# -----------------------------------------------------------------
print("\n=== 2. PANDAS DATAFRAMES ===")

# Create a sample DataFrame mimicking real tabular dataset
data = {
    "age": [25, 30, np.nan, 45, 38],
    "salary": [50000, 64000, 80000, np.nan, 72000],
    "department": ["IT", "HR", "IT", "Sales", "HR"]
}
df = pd.DataFrame(data)
print("Raw DataFrame:\n", df)

# Handling Missing Values
print("\nMissing values per column:\n", df.isna().sum())

# Filling missing numerical values with column median (Standard ML Preprocessing)
df["age"] = df["age"].fillna(df["age"].median())
df["salary"] = df["salary"].fillna(df["salary"].mean())
print("\nImputed DataFrame:\n", df)

# Indexing: loc (label-based) vs iloc (position-based)
print("\nFirst row via iloc[0]:\n", df.iloc[0])
print("\nFiltered rows via loc (age > 30):\n", df.loc[df["age"] > 30, ["age", "department"]])

# GroupBy Aggregation
print("\nAverage Salary per Department:\n", df.groupby("department")["salary"].mean())


# -----------------------------------------------------------------
# 3. BASIC ML MATH (Statistics Essentials)
# -----------------------------------------------------------------
print("\n=== 3. ML MATH & STATISTICS ===")

salaries = df["salary"].values

mean_val = np.mean(salaries)
std_val = np.std(salaries)
var_val = np.var(salaries)

print(f"Mean Salary:              ${mean_val:.2f}")
print(f"Standard Deviation (std): ${std_val:.2f}")
print(f"Variance (var):          ${var_val:.2f}")

# Manual Standard Scaling Formula: z = (x - mean) / std
z_scaled = (salaries - mean_val) / std_val
print("Manually Standardized Salaries (z-scores):\n", z_scaled.round(3))
