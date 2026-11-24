import pandas as pd

df = pd.read_csv("results/cleaned_readings.csv")

print("\n========== SESSION 7 INVESTIGATION ==========")

session = df[df["session_id"] == 7].copy()

print("\nTotal readings:", len(session))

print("\nFirst 20 readings:")

print(
    session[
        [
            "timestamp",
            "voltage_V",
            "current_A",
            "power_W",
            "power_factor",
            "frequency_Hz"
        ]
    ].head(20).to_string(index=False)
)

print("\nCurrent and power combinations:")

print(
    pd.crosstab(
        session["current_A"],
        session["power_W"] > 0
    )
)

print("\nCurrent statistics:")
print(session["current_A"].describe())

print("\nPower statistics:")
print(session["power_W"].describe())

print("\n========== INVESTIGATION COMPLETED ==========")