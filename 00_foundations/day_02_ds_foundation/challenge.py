"""
===================================================================
DAY 02 ACTIVE RECALL CHALLENGE: NumPy, Pandas & ML Math
===================================================================
Instructions:
Try to complete each task below WITHOUT looking back at ds_foundation.py!
"""

import numpy as np
import pandas as pd

# TASK 1: NumPy Reshaping & Dot Product
# 1. Create a 1D numpy array `X_raw` containing values: [1, 2, 3, 4, 5]
# 2. Reshape `X_raw` into a 2D column vector `X` with shape (5, 1)
# 3. Create a weight vector `w = np.array([2.0])` and compute dot product `y_pred = X @ w`


# TASK 2: Pandas Data Cleaning
# Given the raw dataset below:
raw_data = {
    "experience_years": [1, 3, np.nan, 8, 10],
    "rating": [4.2, np.nan, 3.8, 4.9, 4.5]
}
# 1. Create a DataFrame `df`
# 2. Impute missing values in `experience_years` with its median
# 3. Impute missing values in `rating` with its mean
# 4. Filter rows where `experience_years > 4` using `.loc[]`


# TASK 3: Manual Standard Scaling (Z-Score Standardization)
# Given array: `data = np.array([100, 200, 300, 400, 500])`
# Calculate z-score standardized values manually using formula: z = (x - mean) / std


# Write your test code below:
