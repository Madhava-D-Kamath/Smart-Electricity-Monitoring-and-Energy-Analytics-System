import pandas as pd
import numpy as np
import os

# ==========================================
# 1. LOAD ORIGINAL DATASET
# ==========================================

file_path = os.path.join("dataset", "pzem_data.csv")

df = pd.read_csv(file_path)

print("\n========== DATA CLEANING STARTED ==========")

original_count = len(df)

print("Original records:", original_count)

# ==========================================
# 2. STANDARDISE NODE 2 COLUMNS
# ==========================================

# Handle timestamp
if "timestamp" not in df.columns:
    df["timestamp"] = np.nan

if "timestamp_ist" in df.columns:
    df["timestamp"] = df["timestamp"].fillna(
        df["timestamp_ist"]
    )

# Handle node ID
if "node_id" not in df.columns:
    df["node_id"] = np.nan

if "node" in df.columns:
    df["node_id"] = df["node_id"].fillna(df["node"])

# Standardise appliance names
df["appliance"] = df["appliance"].replace({
    "Ceiling / Table Fan": "table fan",
    "ceiling_table_fan": "table fan"
})

# ==========================================
# 3. HANDLE MISSING FREQUENCY
# ==========================================

if "frequency_Hz" not in df.columns:
    df["frequency_Hz"] = np.nan

# Apply assumed nominal frequency only to Node 2
node2_mask = (
    df["node_id"]
    .astype(str)
    .str.lower()
    .str.contains("node2", na=False)
)

df.loc[
    node2_mask & df["frequency_Hz"].isna(),
    "frequency_Hz"
] = 50.0

print("\nNode 2 readings:", node2_mask.sum())

print("Node 2 frequency assumed as 50 Hz:",
      (node2_mask & (df["frequency_Hz"] == 50.0)).sum())

# ==========================================
# 4. CONVERT DATA TYPES
# ==========================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

numeric_columns = [
    "voltage_V",
    "current_A",
    "power_W",
    "energy_kWh",
    "frequency_Hz",
    "power_factor"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# ==========================================
# 5. REMOVE EXACT DUPLICATES
# ==========================================

duplicates = df[df.duplicated(keep="first")].copy()

df = df.drop_duplicates(keep="first").copy()

print("\nExact duplicates removed:", len(duplicates))

# ==========================================
# 6. CHECK MISSING VALUES
# ==========================================

missing_rows = df[
    df.isnull().any(axis=1)
].copy()

print("\nRows containing missing values:",
      len(missing_rows))

# ==========================================
# 7. DETECT ABNORMAL READINGS
# ==========================================

invalid = pd.Series(False, index=df.index)

invalid |= df["timestamp"].isna()

for col in numeric_columns:
    invalid |= df[col].isna()

# Voltage
invalid |= ~df["voltage_V"].between(180, 260)

# Current
invalid |= ~df["current_A"].between(0, 100)

# Power
invalid |= ~df["power_W"].between(0, 26000)

# Frequency
invalid |= ~df["frequency_Hz"].between(45, 55)

# Power factor
invalid |= ~df["power_factor"].between(0, 1)

# Energy
invalid |= df["energy_kWh"] < 0

# Missing appliance labels
invalid |= df["appliance"].isna()

# Save suspicious readings
flagged_data = df[invalid].copy()

# Keep valid readings
cleaned_data = df[~invalid].copy()

print("\nSuspicious readings:", len(flagged_data))

print("Readings passing screening:",
      len(cleaned_data))

# ==========================================
# 8. IDENTIFY MEASUREMENT SESSIONS
# ==========================================

cleaned_data = cleaned_data.sort_values(
    ["node_id", "timestamp"]
).reset_index(drop=True)

cleaned_data["time_difference_sec"] = (
    cleaned_data.groupby("node_id")["timestamp"]
    .diff()
    .dt.total_seconds()
)

previous_appliance = (
    cleaned_data.groupby("node_id")["appliance"]
    .shift(1)
)

new_session = (
    cleaned_data["time_difference_sec"].isna()
    | (cleaned_data["time_difference_sec"] > 60)
    | (cleaned_data["appliance"] != previous_appliance)
)

cleaned_data["session_id"] = new_session.cumsum()

# ==========================================
# 9. CHECK ENERGY COUNTER
# ==========================================

cleaned_data["energy_difference_kWh"] = (
    cleaned_data.groupby("session_id")["energy_kWh"]
    .diff()
)

cleaned_data["energy_counter_decrease"] = (
    cleaned_data["energy_difference_kWh"] < 0
)

energy_decreases = cleaned_data[
    cleaned_data["energy_counter_decrease"]
].copy()

print("\nEnergy counter decreases:",
      len(energy_decreases))

# ==========================================
# 10. CREATE OUTPUT FOLDER
# ==========================================

os.makedirs("results", exist_ok=True)

# ==========================================
# 11. SAVE PROCESSED DATA
# ==========================================

cleaned_data.to_csv(
    "results/cleaned_readings.csv",
    index=False
)

flagged_data.to_csv(
    "results/flagged_readings.csv",
    index=False
)

duplicates.to_csv(
    "results/duplicate_readings.csv",
    index=False
)

missing_rows.to_csv(
    "results/missing_readings.csv",
    index=False
)

energy_decreases.to_csv(
    "results/energy_counter_decreases.csv",
    index=False
)

# ==========================================
# 12. SESSION SUMMARY
# ==========================================

session_summary = (
    cleaned_data.groupby(
        ["session_id", "node_id", "appliance"]
    )
    .agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        total_readings=("timestamp", "count")
    )
    .reset_index()
)

session_summary.to_csv(
    "results/session_summary.csv",
    index=False
)

# ==========================================
# 13. FINAL REPORT
# ==========================================

print("\n========== CLEANING SUMMARY ==========")

print("Original records:", original_count)

print("Duplicates removed:", len(duplicates))

print("Missing value records:", len(missing_rows))

print("Flagged abnormal records:", len(flagged_data))

print("Cleaned records:", len(cleaned_data))

print("Measurement sessions:",
      cleaned_data["session_id"].nunique())

print("Energy counter decreases:",
      len(energy_decreases))

print("\nAppliance-wise cleaned records:")

print(cleaned_data["appliance"].value_counts())

print("\nNode-wise cleaned records:")

print(cleaned_data["node_id"].value_counts())

print("\nAll files saved inside results folder.")

print("\n========== CLEANING COMPLETED ==========")