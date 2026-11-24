import pandas as pd
import numpy as np
import os

# Load cleaned dataset
df = pd.read_csv("results/cleaned_readings.csv")

print("\n========== FEATURE ENGINEERING ==========")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Extract time-based features
df["hour"] = df["timestamp"].dt.hour

df["day"] = df["timestamp"].dt.day

df["day_of_week"] = df["timestamp"].dt.dayofweek

# Cyclical time features
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)

df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Appliance is our target variable
# Electrical measurements are our input features

features = [
    "voltage_V",
    "current_A",
    "power_W",
    "frequency_Hz",
    "power_factor",
    "hour",
    "day_of_week",
    "hour_sin",
    "hour_cos"
]

target = "appliance"

# Create ML-ready dataset
ml_data = df[features + [target, "session_id"]].copy()

# Remove any incomplete feature or target rows
ml_data = ml_data.dropna(subset=features + [target])

# Save dataset
os.makedirs("results", exist_ok=True)

ml_data.to_csv(
    "results/ml_ready_dataset.csv",
    index=False
)

print("\nTotal ML records:", len(ml_data))

print("\nInput features:")
print(features)

print("\nTarget variable:", target)

print("\nAppliance distribution:")
print(ml_data[target].value_counts())

print("\nFinal dataset shape:")
print(ml_data.shape)

print("\nSaved: results/ml_ready_dataset.csv")

print("\n========== FEATURE ENGINEERING COMPLETED ==========")