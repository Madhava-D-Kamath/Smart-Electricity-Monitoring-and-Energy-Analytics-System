
"""
==========================================================
AI SMART ELECTRICITY MONITORING SYSTEM
MODULE: POWER ANALYSIS
==========================================================

Features:
1. Voltage analysis
2. Current analysis
3. Active power analysis
4. Apparent power calculation
5. Reactive power calculation
6. Power factor analysis
7. Energy consumption calculation
8. Electrical measurement statistics
==========================================================
"""

import numpy as np
import pandas as pd


# ==========================================================
# 1. VALIDATE ELECTRICAL DATA
# ==========================================================

REQUIRED_COLUMNS = [
    "voltage_V",
    "current_A",
    "power_W",
    "frequency_Hz",
    "power_factor"
]


def validate_power_data(df):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing electrical columns: {missing}"
        )

    data = df.copy()

    for column in REQUIRED_COLUMNS:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            "Electrical data contains missing or invalid values."
        )

    if (data["voltage_V"] <= 0).any():
        raise ValueError("Voltage must be positive.")

    if (data["current_A"] < 0).any():
        raise ValueError("Current cannot be negative.")

    if (data["power_W"] < 0).any():
        raise ValueError("Power cannot be negative.")

    if (data["frequency_Hz"] <= 0).any():
        raise ValueError("Frequency must be positive.")

    if not data["power_factor"].between(0, 1).all():
        raise ValueError(
            "Power factor must be between 0 and 1."
        )

    return data


# ==========================================================
# 2. SINGLE READING POWER ANALYSIS
# ==========================================================

def analyze_single_reading(reading):
    """
    Analyse one electrical reading.

    Assumes single-phase measurements.
    """

    df = pd.DataFrame([reading])

    data = validate_power_data(df)

    row = data.iloc[0]

    voltage = float(row["voltage_V"])
    current = float(row["current_A"])
    active_power = float(row["power_W"])
    power_factor = float(row["power_factor"])

    # Apparent power
    apparent_power = voltage * current

    # Reactive power magnitude
    reactive_power = np.sqrt(
        max(
            apparent_power ** 2 - active_power ** 2,
            0
        )
    )

    # Power consistency indicator
    expected_power = (
        apparent_power * power_factor
    )

    power_difference = abs(
        active_power - expected_power
    )

    return {
        "voltage_V": voltage,
        "current_A": current,
        "active_power_W": active_power,
        "apparent_power_VA": apparent_power,
        "reactive_power_VAR": float(reactive_power),
        "power_factor": power_factor,
        "frequency_Hz": float(row["frequency_Hz"]),
        "expected_power_W": expected_power,
        "power_difference_W": power_difference
    }


# ==========================================================
# 3. BATCH POWER ANALYSIS
# ==========================================================

def analyze_power_dataset(df):

    data = validate_power_data(df)

    data["apparent_power_VA"] = (
        data["voltage_V"] *
        data["current_A"]
    )

    data["reactive_power_VAR"] = np.sqrt(
        np.maximum(
            data["apparent_power_VA"] ** 2 -
            data["power_W"] ** 2,
            0
        )
    )

    data["expected_power_W"] = (
        data["apparent_power_VA"] *
        data["power_factor"]
    )

    data["power_difference_W"] = abs(
        data["power_W"] -
        data["expected_power_W"]
    )

    return data


# ==========================================================
# 4. ENERGY CONSUMPTION CALCULATION
# ==========================================================

def calculate_energy_kwh(power_w, duration_hours):
    """
    Calculate energy consumption.

    Energy (kWh) = Power (W) * Time (hours) / 1000

    Duration must be provided explicitly.
    """

    power_w = float(power_w)
    duration_hours = float(duration_hours)

    if not np.isfinite(power_w) or power_w < 0:
        raise ValueError("Power must be finite and non-negative.")

    if not np.isfinite(duration_hours) or duration_hours < 0:
        raise ValueError("Duration must be finite and non-negative.")

    return (
        power_w * duration_hours / 1000
    )


# ==========================================================
# 5. CUMULATIVE ENERGY ANALYSIS
# ==========================================================

def calculate_energy_from_readings(
    df,
    timestamp_column="timestamp"
):
    """
    Estimate energy from timestamped power readings.

    Uses trapezoidal integration between consecutive
    timestamped readings.

    This assumes the power readings represent the
    intervals between measurements reasonably well.
    """

    required = [
        timestamp_column,
        "power_W"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data = df[required].copy()

    data[timestamp_column] = pd.to_datetime(
        data[timestamp_column],
        errors="coerce",
        utc=True
    )

    data["power_W"] = pd.to_numeric(
        data["power_W"],
        errors="coerce"
    )

    data = data.dropna(
        subset=required
    )

    data = data.sort_values(
        timestamp_column
    ).reset_index(drop=True)

    if len(data) < 2:
        data["energy_interval_kWh"] = []
        data["cumulative_energy_kWh"] = []
        return data

    if (data["power_W"] < 0).any():
        raise ValueError(
            "Power cannot be negative."
        )

    time_difference_hours = (
        data[timestamp_column]
        .diff()
        .dt.total_seconds() / 3600
    )

    power_average = (
        data["power_W"] +
        data["power_W"].shift(1)
    ) / 2

    data["energy_interval_kWh"] = (
        power_average *
        time_difference_hours / 1000
    )

    data["energy_interval_kWh"] = (
        data["energy_interval_kWh"].fillna(0)
    )

    if (time_difference_hours.dropna() < 0).any():
        raise ValueError(
            "Timestamps must be in chronological order."
        )

    data["cumulative_energy_kWh"] = (
        data["energy_interval_kWh"].cumsum()
    )

    return data


# ==========================================================
# 6. POWER STATISTICS
# ==========================================================

def get_power_statistics(df):

    data = validate_power_data(df)

    statistics = {}

    columns = [
        "voltage_V",
        "current_A",
        "power_W",
        "frequency_Hz",
        "power_factor"
    ]

    for column in columns:

        statistics[column] = {
            "minimum": float(data[column].min()),
            "maximum": float(data[column].max()),
            "average": float(data[column].mean()),
            "standard_deviation": float(
                data[column].std()
            ) if len(data) > 1 else 0.0
        }

    statistics["total_readings"] = len(data)

    return statistics


# ==========================================================
# 7. POWER FACTOR ANALYSIS
# ==========================================================

def analyze_power_factor(power_factor):

    power_factor = float(power_factor)

    if not np.isfinite(power_factor) or not 0 <= power_factor <= 1:
        raise ValueError(
            "Power factor must be between 0 and 1."
        )

    if power_factor >= 0.95:
        status = "High power factor"

    elif power_factor >= 0.80:
        status = "Moderate power factor"

    else:
        status = "Low power factor"

    return {
        "power_factor": power_factor,
        "status": status
    }
