
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


REQUIRED_COLUMNS = ["timestamp", "appliance", "power_W"]


def validate_historical_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare historical electricity readings."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Historical dataset is empty.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )
    data["power_W"] = pd.to_numeric(
        data["power_W"], errors="coerce"
    )

    data = data.dropna(
        subset=["timestamp", "appliance", "power_W"]
    )

    data = data[data["power_W"] >= 0]

    if data.empty:
        raise ValueError("No valid historical readings were found.")

    return data.sort_values("timestamp").reset_index(drop=True)


def power_consumption_chart(df: pd.DataFrame):
    """Display power consumption over time."""

    data = validate_historical_data(df)

    fig = px.line(
        data,
        x="timestamp",
        y="power_W",
        color="appliance",
        title="Historical Power Consumption",
        labels={
            "timestamp": "Time",
            "power_W": "Power (W)",
            "appliance": "Appliance"
        }
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


def appliance_comparison_chart(df: pd.DataFrame):
    """Compare average power consumption by appliance."""

    data = validate_historical_data(df)

    summary = (
        data.groupby("appliance")["power_W"]
        .mean()
        .reset_index(name="average_power_W")
        .sort_values("average_power_W", ascending=False)
    )

    fig = px.bar(
        summary,
        x="appliance",
        y="average_power_W",
        title="Average Power Consumption by Appliance",
        labels={
            "appliance": "Appliance",
            "average_power_W": "Average Power (W)"
        },
        text_auto=".2f"
    )

    fig.update_layout(template="plotly_white")

    return fig


def appliance_distribution_chart(df: pd.DataFrame):
    """Show the distribution of recorded readings by appliance."""

    data = validate_historical_data(df)

    summary = (
        data.groupby("appliance")
        .size()
        .reset_index(name="reading_count")
    )

    fig = px.pie(
        summary,
        names="appliance",
        values="reading_count",
        title="Appliance Reading Distribution",
        hole=0.4
    )

    fig.update_layout(template="plotly_white")

    return fig


def daily_power_chart(df: pd.DataFrame):
    """Display daily average power readings."""

    data = validate_historical_data(df)

    data["date"] = data["timestamp"].dt.date

    daily = (
        data.groupby("date")["power_W"]
        .mean()
        .reset_index(name="average_power_W")
    )

    fig = px.line(
        daily,
        x="date",
        y="average_power_W",
        markers=True,
        title="Daily Average Power Consumption",
        labels={
            "date": "Date",
            "average_power_W": "Average Power (W)"
        }
    )

    fig.update_layout(template="plotly_white")

    return fig


def monthly_power_chart(df: pd.DataFrame):
    """Display monthly average power readings."""

    data = validate_historical_data(df)

    data["month"] = (
        data["timestamp"].dt.to_period("M").astype(str)
    )

    monthly = (
        data.groupby("month")["power_W"]
        .mean()
        .reset_index(name="average_power_W")
    )

    fig = px.bar(
        monthly,
        x="month",
        y="average_power_W",
        title="Monthly Average Power Consumption",
        labels={
            "month": "Month",
            "average_power_W": "Average Power (W)"
        },
        text_auto=".2f"
    )

    fig.update_layout(template="plotly_white")

    return fig


def calculate_historical_energy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate energy between consecutive readings.

    Uses trapezoidal integration of power over elapsed time.
    Readings are grouped by appliance and session_id when available.
    """

    data = validate_historical_data(df)

    group_columns = ["appliance"]

    if "session_id" in data.columns:
        group_columns.append("session_id")

    results = []

    for _, group in data.groupby(
        group_columns, dropna=False, sort=False
    ):
        group = group.sort_values("timestamp").copy()

        time_hours = (
            group["timestamp"].diff().dt.total_seconds() / 3600
        )

        previous_power = group["power_W"].shift(1)

        energy = (
            (previous_power + group["power_W"]) / 2
        ) * time_hours / 1000

        group["energy_interval_kWh"] = energy.fillna(0)

        # Do not integrate across long gaps in recorded data.
        gap_hours = time_hours
        group.loc[
            gap_hours > 24, "energy_interval_kWh"
        ] = 0

        results.append(group)

    return pd.concat(results, ignore_index=True)


def historical_energy_chart(df: pd.DataFrame):
    """Display estimated energy consumption by appliance."""

    data = calculate_historical_energy(df)

    summary = (
        data.groupby("appliance")["energy_interval_kWh"]
        .sum()
        .reset_index()
        .sort_values("energy_interval_kWh", ascending=False)
    )

    fig = px.bar(
        summary,
        x="appliance",
        y="energy_interval_kWh",
        title="Estimated Historical Energy by Appliance",
        labels={
            "appliance": "Appliance",
            "energy_interval_kWh": "Estimated Energy (kWh)"
        },
        text_auto=".4f"
    )

    fig.update_layout(template="plotly_white")

    return fig


def peak_power_analysis(df: pd.DataFrame) -> dict:
    """Identify the highest recorded power reading."""

    data = validate_historical_data(df)

    peak = data.loc[data["power_W"].idxmax()]

    return {
        "peak_power_W": float(peak["power_W"]),
        "appliance": str(peak["appliance"]),
        "timestamp": str(peak["timestamp"]),
        "average_power_W": float(data["power_W"].mean()),
        "minimum_power_W": float(data["power_W"].min()),
        "total_readings": int(len(data))
    }


def generate_visualization_report(df: pd.DataFrame) -> dict:
    """Generate all historical visualization figures and statistics."""

    data = validate_historical_data(df)

    return {
        "power_trend": power_consumption_chart(data),
        "appliance_comparison": appliance_comparison_chart(data),
        "appliance_distribution": appliance_distribution_chart(data),
        "daily_power": daily_power_chart(data),
        "monthly_power": monthly_power_chart(data),
        "historical_energy": historical_energy_chart(data),
        "peak_analysis": peak_power_analysis(data)
    }
