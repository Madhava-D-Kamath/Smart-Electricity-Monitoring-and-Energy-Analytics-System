import pandas as pd

# Load existing master dataset
old_df = pd.read_csv("dataset/pzem_data.csv")

# Load new Node 2 dataset
new_df = pd.read_csv("dataset/node2_ceiling_table_fan_2026-11-21.csv")

# Rename columns to match master dataset
new_df = new_df.rename(columns={
    "voltage": "voltage_V",
    "current": "current_A",
    "power": "power_W",
    "energy": "energy_kWh",
    "power_factor": "power_factor"
})

# Add missing frequency column
if "frequency_Hz" not in new_df.columns:
    new_df["frequency_Hz"] = float("nan")

# Combine datasets
combined_df = pd.concat(
    [old_df, new_df],
    ignore_index=True
)

# Remove exact duplicates
combined_df.drop_duplicates(inplace=True)

# Save updated master dataset
combined_df.to_csv("dataset/pzem_data.csv", index=False)

print("Dataset updated successfully!")
print("Old readings:", len(old_df))
print("New readings:", len(new_df))
print("Total readings:", len(combined_df))