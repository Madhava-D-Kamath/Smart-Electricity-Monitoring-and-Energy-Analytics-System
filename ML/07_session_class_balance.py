import pandas as pd

df = pd.read_csv("results/ml_ready_dataset.csv")

print("\n========== SESSION CLASS BALANCE ==========")

# Number of sessions per appliance
session_counts = (
    df.groupby("appliance")["session_id"]
    .nunique()
)

print("\nSessions per appliance:")
print(session_counts)

# Readings in each session
session_summary = (
    df.groupby(["session_id", "appliance"])
    .agg(
        total_readings=("power_W", "count"),
        average_power=("power_W", "mean"),
        minimum_power=("power_W", "min"),
        maximum_power=("power_W", "max")
    )
    .reset_index()
)

print("\nIndividual session details:")
print(session_summary.to_string(index=False))

# Save summary
session_summary.to_csv(
    "results/session_class_balance.csv",
    index=False
)

print("\nSaved: results/session_class_balance.csv")

print("\n========== ANALYSIS COMPLETED ==========")