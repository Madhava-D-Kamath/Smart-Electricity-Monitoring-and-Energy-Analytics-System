import pandas as pd
import os

# Dataset location
file_path = os.path.join("dataset", "pzem_data.csv")

# Load dataset
df = pd.read_csv(file_path)

print("\n========== DATASET INFORMATION ==========")

# Display first 5 records
print("\nFirst 5 Records:")
print(df.head())

# Dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Statistical information
print("\nStatistical Summary:")
print(df.describe())

print("\n========== INSPECTION COMPLETED ==========")