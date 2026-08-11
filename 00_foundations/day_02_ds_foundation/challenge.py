# """
# DAY 02 ACTIVE RECALL CHALLENGE: NumPy, Pandas & ML Math

import numpy as np
import pandas as pd


# TASK 1: NumPy Reshaping & Dot Product

# 1. Create a 1D numpy array `X_raw` containing values: [1, 2, 3, 4, 5]

X_raw = np.array([1, 2, 3, 4, 5])

print("X_raw:", X_raw)


# 2. Reshape `X_raw` into a 2D column vector `X` with shape (5, 1)

X = X_raw.reshape(5, 1)

print("X:")
print(X)

print("Shape:", X.shape)


# 3. Create a weight vector `w = np.array([2.0])`
# and compute dot product `y_pred = X @ w`

w = np.array([2.0])

y_pred = X @ w

print("Weight:", w)
print("Predictions:")
print(y_pred)


# TASK 2: Pandas Data Cleaning

# Given the raw dataset below:

raw_data = {
    "experience_years": [1, 3, np.nan, 8, 10],
    "rating": [4.2, np.nan, 3.8, 4.9, 4.5]
}


# 1. Create a DataFrame `df`

df = pd.DataFrame(raw_data)

print("\nOriginal DataFrame:")
print(df)


# 2. Impute missing values in `experience_years` with its median

df["experience_years"] = df["experience_years"].fillna(
    df["experience_years"].median()
)


# 3. Impute missing values in `rating` with its mean

df["rating"] = df["rating"].fillna(
    df["rating"].mean()
)

print("\nDataFrame after filling missing values:")
print(df)


# 4. Filter rows where `experience_years > 4` using `.loc[]`

filtered_df = df.loc[df["experience_years"] > 4]

print("\nRows where experience_years > 4:")
print(filtered_df)


# TASK 3: Manual Standard Scaling (Z-Score Standardization)

# Given array:
# `data = np.array([100, 200, 300, 400, 500])`

data = np.array([100, 200, 300, 400, 500])


# Calculate z-score standardized values manually
# using formula: z = (x - mean) / std

mean_val = np.mean(data)
std_val = np.std(data)

z_scaled = (data - mean_val) / std_val

print("\nMean:", mean_val)
print("Standard Deviation:", std_val)
print("Z-Scores:", z_scaled.round(3))