import pandas as pd
import os

# Load dataset
file_path = os.path.join("dataset", "pzem_data.csv")
df = pd.read_csv(file_path)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)

print("\n========== SESSION ANALYSIS ==========")

# 1. Overall dataset information
print("\nTotal readings:", len(df))

print("\nAppliance reading counts:")
print(df["appliance"].value_counts())

# 2. Time range for each appliance
print("\nAppliance measurement periods:")

for appliance, group in df.groupby("appliance"):
    print("\nAppliance:", appliance)
    print("First reading:", group["timestamp"].min())
    print("Last reading:", group["timestamp"].max())
    print("Number of readings:", len(group))

# 3. Calculate time differences
df["time_difference"] = df["timestamp"].diff().dt.total_seconds()

print("\nTime difference statistics:")
print(df["time_difference"].describe())

# 4. Identify large gaps
gaps = df[df["time_difference"] > 60]

print("\nGaps greater than 60 seconds:")
print(gaps[[
    "timestamp",
    "appliance",
    "time_difference"
]].head(30))

print("\nTotal large gaps:", len(gaps))

# 5. Identify appliance transitions
df["previous_appliance"] = df["appliance"].shift(1)

transitions = df[
    df["appliance"] != df["previous_appliance"]
]

print("\nAppliance transitions:")
print(transitions[[
    "timestamp",
    "previous_appliance",
    "appliance"
]].to_string(index=False))

# 6. Examine cumulative energy
df["energy_difference"] = df["energy_kWh"].diff()

print("\nEnergy counter statistics:")
print(df["energy_kWh"].describe())

print("\nEnergy difference statistics:")
print(df["energy_difference"].describe())

# 7. Check energy counter decreases
decreases = df[df["energy_difference"] < 0]

print("\nEnergy counter decreases:")
print(decreases[[
    "timestamp",
    "appliance",
    "energy_kWh",
    "energy_difference"
]])

# 8. Save chronological analysis
os.makedirs("results", exist_ok=True)

df.to_csv(
    "results/session_analysis.csv",
    index=False
)

print("\nAnalysis saved to results/session_analysis.csv")

print("\n========== ANALYSIS COMPLETED ==========")