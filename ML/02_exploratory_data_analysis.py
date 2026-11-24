import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
file_path = os.path.join("dataset", "pzem_data.csv")

df = pd.read_csv(file_path)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Remove the clearly invalid extreme sensor record
# We will preserve the original CSV.
df = df[
    (df["voltage_V"].between(180, 260)) &
    (df["current_A"].between(0, 100)) &
    (df["power_W"].between(0, 26000)) &
    (df["frequency_Hz"].between(45, 55)) &
    (df["power_factor"].between(0, 1))
].copy()

# Create results directory
os.makedirs("results", exist_ok=True)

# 1. Voltage distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["voltage_V"], bins=30, kde=True)
plt.title("Voltage Distribution")
plt.xlabel("Voltage (V)")
plt.ylabel("Frequency")
plt.savefig("results/voltage_distribution.png")
plt.show()

# 2. Current distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["current_A"], bins=30, kde=True)
plt.title("Current Distribution")
plt.xlabel("Current (A)")
plt.ylabel("Frequency")
plt.savefig("results/current_distribution.png")
plt.show()

# 3. Power consumption by appliance
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="appliance", y="power_W")
plt.title("Power Consumption by Appliance")
plt.xlabel("Appliance")
plt.ylabel("Power (W)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("results/appliance_power.png")
plt.show()

# 4. Energy distribution by appliance
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="appliance", y="energy_kWh")
plt.title("Cumulative Energy by Appliance")
plt.xlabel("Appliance")
plt.ylabel("Energy (kWh)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("results/appliance_energy.png")
plt.show()

# 5. Power consumption over time
plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["power_W"])
plt.title("Power Consumption Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Power (W)")
plt.tight_layout()
plt.savefig("results/power_over_time.png")
plt.show()

print("EDA completed successfully!")
print("Graphs saved in results folder.")