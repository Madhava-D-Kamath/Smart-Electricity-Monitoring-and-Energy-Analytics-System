
"""
==========================================================
AI SMART ELECTRICITY MONITORING SYSTEM
MODULE: APPLIANCE-WISE ENERGY ANALYTICS
==========================================================

Features:
1. Appliance-wise energy consumption
2. Total energy consumption
3. Average power analysis
4. Maximum and minimum power
5. Appliance consumption comparison
6. Energy contribution percentage
7. Daily energy summary
8. Monthly energy summary

Required columns:
- timestamp
- appliance
- power_W

Optional:
- energy_kWh (cumulative meter reading)
==========================================================
"""

import numpy as np
import pandas as pd


# ==========================================================
# 1. DATA VALIDATION
# ==========================================================

def validate_energy_data(df):

    required_columns = [
        "timestamp",
        "appliance",
        "power_W"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data = df.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
        utc=True
    )

    data["power_W"] = pd.to_numeric(
        data["power_W"],
        errors="coerce"
    )

    data["appliance"] = (
        data["appliance"]
        .astype("string")
        .str.strip()
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna(
        subset=[
            "timestamp",
            "appliance",
            "power_W"
        ]
    )

    data = data[
        (data["appliance"] != "") &
        (data["power_W"] >= 0)
    ].copy()

    return data


# ==========================================================
# 2. CALCULATE ENERGY FROM POWER READINGS
# ==========================================================

def calculate_energy_intervals(df):

    """
    Estimate energy using trapezoidal integration.

    Energy (kWh) = Average Power (W) * Time (hours) / 1000

    Calculations are performed independently for
    each appliance and session when session_id exists.

    This requires sufficiently frequent readings.
    """

    data = validate_energy_data(df)

    if data.empty:
        return data

    group_columns = ["appliance"]

    if "session_id" in data.columns:

        data["session_id"] = (
            data["session_id"]
            .astype("string")
            .str.strip()
        )

        group_columns.append("session_id")

    data = data.sort_values(
        group_columns + ["timestamp"]
    ).reset_index(drop=True)

    def calculate_group(group):

        group = group.copy()

        time_hours = (
            group["timestamp"]
            .diff()
            .dt.total_seconds() / 3600
        )

        average_power = (
            group["power_W"] +
            group["power_W"].shift(1)
        ) / 2

        energy = (
            average_power * time_hours / 1000
        )

        # First reading has no preceding interval.
        # Negative time intervals are invalid.
        energy = energy.where(
            time_hours >= 0,
            np.nan
        )

        group["energy_interval_kWh"] = (
            energy.fillna(0)
        )

        group["cumulative_energy_kWh"] = (
            group["energy_interval_kWh"].cumsum()
        )

        return group

    data = (
        data.groupby(
            group_columns,
            group_keys=False
        )
        .apply(
            calculate_group,
            include_groups=False
        )
        .reset_index(drop=True)
    )

    return data


# ==========================================================
# 3. APPLIANCE-WISE ENERGY SUMMARY
# ==========================================================

def appliance_energy_summary(df):

    data = calculate_energy_intervals(df)

    if data.empty:
        return pd.DataFrame()

    summary = (
        data.groupby("appliance")
        .agg(
            total_energy_kWh=(
                "energy_interval_kWh",
                "sum"
            ),
            average_power_W=(
                "power_W",
                "mean"
            ),
            maximum_power_W=(
                "power_W",
                "max"
            ),
            minimum_power_W=(
                "power_W",
                "min"
            ),
            total_readings=(
                "power_W",
                "count"
            )
        )
        .reset_index()
    )

    total_energy = summary[
        "total_energy_kWh"
    ].sum()

    if total_energy > 0:

        summary["energy_contribution_percent"] = (
            summary["total_energy_kWh"] /
            total_energy * 100
        )

    else:

        summary["energy_contribution_percent"] = 0.0

    return summary.sort_values(
        "total_energy_kWh",
        ascending=False
    ).reset_index(drop=True)


# ==========================================================
# 4. TOTAL ENERGY CONSUMPTION
# ==========================================================

def calculate_total_energy(df):

    data = calculate_energy_intervals(df)

    if data.empty:
        return 0.0

    return float(
        data["energy_interval_kWh"].sum()
    )


# ==========================================================
# 5. DAILY ENERGY SUMMARY
# ==========================================================

def daily_energy_summary(df):

    data = calculate_energy_intervals(df)

    if data.empty:
        return pd.DataFrame()

    data["date"] = (
        data["timestamp"]
        .dt.tz_convert("Asia/Kolkata")
        .dt.date
    )

    daily = (
        data.groupby("date")
        .agg(
            energy_kWh=(
                "energy_interval_kWh",
                "sum"
            )
        )
        .reset_index()
    )

    return daily


# ==========================================================
# 6. MONTHLY ENERGY SUMMARY
# ==========================================================

def monthly_energy_summary(df):

    data = calculate_energy_intervals(df)

    if data.empty:
        return pd.DataFrame()

    data["month"] = (
        data["timestamp"]
        .dt.tz_convert("Asia/Kolkata")
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        data.groupby("month")
        .agg(
            energy_kWh=(
                "energy_interval_kWh",
                "sum"
            )
        )
        .reset_index()
    )

    return monthly


# ==========================================================
# 7. TOP ENERGY-CONSUMING APPLIANCES
# ==========================================================

def top_energy_consumers(df, top_n=5):

    if top_n < 1:
        raise ValueError(
            "top_n must be at least 1."
        )

    summary = appliance_energy_summary(df)

    return summary.head(top_n)


# ==========================================================
# 8. ENERGY ANALYTICS REPORT
# ==========================================================

def generate_energy_report(df):

    data = calculate_energy_intervals(df)

    if data.empty:

        return {
            "total_energy_kWh": 0.0,
            "total_readings": 0,
            "appliance_count": 0,
            "average_power_W": 0.0,
            "maximum_power_W": 0.0,
            "appliance_summary": pd.DataFrame()
        }

    summary = appliance_energy_summary(df)

    return {
        "total_energy_kWh": float(
            data["energy_interval_kWh"].sum()
        ),
        "total_readings": len(data),
        "appliance_count": data["appliance"].nunique(),
        "average_power_W": float(
            data["power_W"].mean()
        ),
        "maximum_power_W": float(
            data["power_W"].max()
        ),
        "appliance_summary": summary
    }
