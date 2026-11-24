import pandas as pd

df = pd.read_csv("results/cleaned_readings.csv")

print("\n========== UNUSUAL SESSION ANALYSIS ==========")

# Examine mobile charging sessions
print("\nMOBILE CHARGING SESSIONS")

mobile = df[df["appliance"] == "mobile charging"]

mobile_summary = (
    mobile.groupby("session_id")
    .agg(
        readings=("power_W", "count"),
        min_power=("power_W", "min"),
        max_power=("power_W", "max"),
        avg_power=("power_W", "mean"),
        avg_current=("current_A", "mean"),
        avg_voltage=("voltage_V", "mean"),
        avg_pf=("power_factor", "mean")
    )
)

print(mobile_summary.round(3))

# Examine iron box and kettle
print("\nIRON BOX AND KETTLE")

heating = df[
    df["appliance"].isin(["iron box", "kettle"])
]

heating_summary = (
    heating.groupby(["session_id", "appliance"])
    .agg(
        readings=("power_W", "count"),
        min_power=("power_W", "min"),
        max_power=("power_W", "max"),
        avg_power=("power_W", "mean"),
        zero_readings=("power_W", lambda x: (x == 0).sum())
    )
)

print(heating_summary.round(3))

# Save results
mobile_summary.to_csv(
    "results/mobile_session_investigation.csv"
)

heating_summary.to_csv(
    "results/heating_session_investigation.csv"
)

print("\nInvestigation files saved.")

print("\n========== ANALYSIS COMPLETED ==========")