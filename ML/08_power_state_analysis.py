import pandas as pd

df = pd.read_csv("results/cleaned_readings.csv")

print("\n========== POWER STATE ANALYSIS ==========")

# Identify zero-power readings
df["power_state"] = df["power_W"].apply(
    lambda x: "Zero power" if x == 0 else "Active power"
)

# Appliance-wise power states
summary = pd.crosstab(
    df["appliance"],
    df["power_state"]
)

print("\nAppliance-wise power states:")
print(summary)

# Calculate percentage of zero-power readings
zero_percentage = (
    df[df["power_W"] == 0]
    .groupby("appliance")
    .size()
    .div(df.groupby("appliance").size())
    * 100
)

print("\nZero-power percentage:")
print(zero_percentage.round(2))

# Session-wise analysis
session_summary = (
    df.groupby(["session_id", "appliance"])
    .agg(
        readings=("power_W", "count"),
        zero_power_readings=(
            "power_W",
            lambda x: (x == 0).sum()
        ),
        average_power=("power_W", "mean")
    )
    .reset_index()
)

print("\nSession-wise power states:")
print(session_summary.to_string(index=False))

summary.to_csv(
    "results/power_state_summary.csv"
)

session_summary.to_csv(
    "results/session_power_analysis.csv",
    index=False
)

print("\nAnalysis files saved successfully.")

print("\n========== ANALYSIS COMPLETED ==========")