from __future__ import annotations

from typing import Any

import pandas as pd


REQUIRED_COLUMNS = ["appliance", "power_W"]


def validate_recommendation_data(df: pd.DataFrame) -> None:
    """Validate the dataset before generating recommendations."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The dataset is empty.")

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["appliance"].isna().any():
        raise ValueError("Appliance names cannot contain missing values.")

    if df["power_W"].isna().any():
        raise ValueError("Power readings cannot contain missing values.")

    if (pd.to_numeric(df["power_W"], errors="coerce").isna()).any():
        raise ValueError("Power readings must be numeric.")

    if (pd.to_numeric(df["power_W"]) < 0).any():
        raise ValueError("Power readings cannot be negative.")


def appliance_consumption_summary(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Summarize average, maximum and total recorded power by appliance."""

    validate_recommendation_data(df)

    data = df.copy()
    data["power_W"] = pd.to_numeric(data["power_W"])

    summary = (
        data.groupby("appliance")["power_W"]
        .agg(
            average_power_W="mean",
            maximum_power_W="max",
            total_recorded_power_W="sum",
            reading_count="count"
        )
        .reset_index()
    )

    return summary.sort_values(
        by="total_recorded_power_W",
        ascending=False
    ).reset_index(drop=True)


def generate_energy_recommendations(
    df: pd.DataFrame,
    high_power_threshold_W: float = 1000.0,
    low_power_factor_threshold: float = 0.85
) -> pd.DataFrame:
    """
    Generate recommendations using measured power and power factor.

    Thresholds are configurable screening values, not universal
    definitions of inefficient or faulty appliances.
    """

    validate_recommendation_data(df)

    if high_power_threshold_W <= 0:
        raise ValueError("High power threshold must be positive.")

    if not 0 < low_power_factor_threshold <= 1:
        raise ValueError("Power factor threshold must be between 0 and 1.")

    data = df.copy()
    data["power_W"] = pd.to_numeric(data["power_W"])

    recommendations = []

    summary = appliance_consumption_summary(data)

    if len(summary) > 0:
        highest_consumer = summary.iloc[0]

        recommendations.append({
            "category": "Consumption Analysis",
            "appliance": highest_consumer["appliance"],
            "priority": "Information",
            "observation": (
                f"{highest_consumer['appliance']} has the highest "
                "total recorded power among the appliances."
            ),
            "recommendation": (
                "Review its operating duration and usage pattern "
                "to identify possible energy-saving opportunities."
            )
        })

    high_power = data[data["power_W"] > high_power_threshold_W]

    for appliance in high_power["appliance"].unique():
        recommendations.append({
            "category": "High Power Reading",
            "appliance": appliance,
            "priority": "Review",
            "observation": (
                f"{appliance} has readings above "
                f"{high_power_threshold_W:.0f} W."
            ),
            "recommendation": (
                "Check whether this power level is expected for "
                "the appliance and avoid unnecessary operating time."
            )
        })

    if "power_factor" in data.columns:
        data["power_factor"] = pd.to_numeric(
            data["power_factor"],
            errors="coerce"
        )

        valid_pf = data[
            data["power_factor"].notna()
            & (data["power_factor"] > 0)
            & (data["power_factor"] <= 1)
        ]

        low_pf = valid_pf[
            valid_pf["power_factor"] < low_power_factor_threshold
        ]

        for appliance in low_pf["appliance"].unique():
            average_pf = low_pf.loc[
                low_pf["appliance"] == appliance,
                "power_factor"
            ].mean()

            recommendations.append({
                "category": "Power Factor",
                "appliance": appliance,
                "priority": "Review",
                "observation": (
                    f"Recorded power factor is below "
                    f"{low_power_factor_threshold:.2f}."
                ),
                "recommendation": (
                    "Review the appliance's electrical characteristics "
                    "and power-factor readings."
                )
            })

    if not recommendations:
        recommendations.append({
            "category": "General",
            "appliance": "All Appliances",
            "priority": "Information",
            "observation": "No conditions matched the configured screening rules.",
            "recommendation": (
                "Continue monitoring consumption and collect more "
                "readings to identify usage patterns."
            )
        })

    return pd.DataFrame(recommendations)


def estimate_savings_potential(
    power_W: float,
    reduced_hours_per_day: float,
    electricity_rate_per_kWh: float
) -> dict[str, float]:
    """
    Estimate savings for a hypothetical reduction in operating time.

    This is a scenario estimate, not a guaranteed saving.
    """

    if power_W < 0:
        raise ValueError("Power cannot be negative.")

    if reduced_hours_per_day < 0:
        raise ValueError("Reduced hours cannot be negative.")

    if electricity_rate_per_kWh < 0:
        raise ValueError("Electricity rate cannot be negative.")

    energy_saved_kWh = (
        power_W * reduced_hours_per_day
    ) / 1000

    daily_savings = (
        energy_saved_kWh * electricity_rate_per_kWh
    )

    return {
        "energy_saved_kWh_per_day": round(energy_saved_kWh, 4),
        "estimated_daily_savings": round(daily_savings, 2),
        "estimated_monthly_savings": round(daily_savings * 30, 2)
    }


def generate_recommendation_report(
    df: pd.DataFrame
) -> dict[str, Any]:
    """Generate a combined appliance summary and recommendation report."""

    summary = appliance_consumption_summary(df)
    recommendations = generate_energy_recommendations(df)

    return {
        "appliance_summary": summary,
        "recommendations": recommendations,
        "total_appliances": int(summary["appliance"].nunique()),
        "total_readings": int(summary["reading_count"].sum())
    }