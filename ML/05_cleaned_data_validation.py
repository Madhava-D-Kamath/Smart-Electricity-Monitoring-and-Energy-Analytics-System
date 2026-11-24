import pandas as pd

df = pd.read_csv("results/cleaned_readings.csv")

print("\n========== CLEANED DATA VALIDATION ==========")

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

print("\nAppliance distribution:")
print(df["appliance"].value_counts())

print("\nVoltage range:")
print(df["voltage_V"].min(), "to", df["voltage_V"].max())

print("\nCurrent range:")
print(df["current_A"].min(), "to", df["current_A"].max())

print("\nPower range:")
print(df["power_W"].min(), "to", df["power_W"].max())

print("\nFrequency range:")
print(df["frequency_Hz"].min(), "to", df["frequency_Hz"].max())

print("\nPower factor range:")
print(df["power_factor"].min(), "to", df["power_factor"].max())

print("\nSession distribution:")
print(df.groupby("session_id")["appliance"].first())

print("\n========== VALIDATION COMPLETED ==========")