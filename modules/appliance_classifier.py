"""
==========================================================
AI SMART ELECTRICITY MONITORING SYSTEM
MODULE: APPLIANCE CLASSIFICATION
==========================================================

Features:
1. Electrical measurement validation
2. ML-based appliance identification
3. Prediction probability calculation
4. Model confidence interpretation
5. Top appliance predictions
6. Support for multiple appliance classes

Compatible with:
- Random Forest
- XGBoost
- Scikit-learn classifiers
- Joblib saved models
==========================================================
"""

import numpy as np
import pandas as pd


# ==========================================================
# 1. MODEL FEATURES
# ==========================================================

FEATURES = [
    "voltage_V",
    "current_A",
    "power_W",
    "frequency_Hz",
    "power_factor"
]


# ==========================================================
# 2. VALIDATE SENSOR READING
# ==========================================================

def validate_reading(reading):
    """
    Validate electrical measurements before prediction.

    Parameters:
        reading (dict): Electrical sensor measurements.

    Returns:
        dict: Validated numerical measurements.
    """

    if not isinstance(reading, dict):
        raise TypeError("Reading must be provided as a dictionary.")

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in reading
    ]

    if missing_features:
        raise ValueError(
            f"Missing electrical features: {missing_features}"
        )

    values = {}

    for feature in FEATURES:

        try:
            value = float(reading[feature])

        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid numeric value for {feature}"
            )

        if not np.isfinite(value):
            raise ValueError(
                f"Non-finite value detected for {feature}"
            )

        values[feature] = value

    # Electrical measurement validation

    if values["voltage_V"] <= 0:
        raise ValueError("Voltage must be greater than zero.")

    if values["current_A"] < 0:
        raise ValueError("Current cannot be negative.")

    if values["power_W"] < 0:
        raise ValueError("Power cannot be negative.")

    if values["frequency_Hz"] <= 0:
        raise ValueError("Frequency must be greater than zero.")

    if not 0 <= values["power_factor"] <= 1:
        raise ValueError(
            "Power factor must be between 0 and 1."
        )

    return values


# ==========================================================
# 3. PREPARE MODEL INPUT
# ==========================================================

def prepare_model_input(reading):
    """
    Convert sensor readings into a DataFrame
    compatible with the trained ML model.
    """

    validated_reading = validate_reading(reading)

    input_df = pd.DataFrame(
        [validated_reading],
        columns=FEATURES
    )

    return input_df


# ==========================================================
# 4. APPLIANCE CLASSIFICATION
# ==========================================================

def classify_appliance(model, reading):
    """
    Identify the appliance using a trained ML model.

    Parameters:
        model: Trained classification model.
        reading: Electrical sensor readings.

    Returns:
        Dictionary containing:
            prediction
            confidence
            probabilities
            status
    """

    input_df = prepare_model_input(reading)

    # Predict appliance
    prediction = model.predict(input_df)[0]

    probabilities = {}
    confidence = None

    # Calculate class probabilities
    if hasattr(model, "predict_proba"):

        scores = model.predict_proba(input_df)[0]

        classes = model.classes_

        probabilities = {
            str(class_name): float(score)
            for class_name, score in zip(classes, scores)
        }

        confidence = max(probabilities.values()) * 100

    # Determine prediction status
    if confidence is None:

        status = "Probability unavailable"

    elif confidence >= 85:

        status = "High model confidence"

    elif confidence >= 60:

        status = "Moderate model confidence"

    else:

        status = "Low model confidence"

    return {
        "prediction": str(prediction),
        "confidence": confidence,
        "probabilities": probabilities,
        "status": status
    }


# ==========================================================
# 5. TOP APPLIANCE PREDICTIONS
# ==========================================================

def get_top_predictions(model, reading, top_n=3):
    """
    Return the top N appliance predictions.

    Useful for displaying alternative predictions
    in the Streamlit dashboard.
    """

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    input_df = prepare_model_input(reading)

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "The selected model does not support predict_proba()."
        )

    scores = model.predict_proba(input_df)[0]

    classes = model.classes_

    results = [
        {
            "appliance": str(class_name),
            "probability": float(score) * 100
        }
        for class_name, score in zip(classes, scores)
    ]

    results.sort(
        key=lambda item: item["probability"],
        reverse=True
    )

    return results[:top_n]


# ==========================================================
# 6. CLASSIFICATION SUMMARY
# ==========================================================

def get_classification_summary(model, reading):
    """
    Generate a complete appliance classification summary.
    """

    result = classify_appliance(
        model,
        reading
    )

    summary = {
        "Detected Appliance": result["prediction"],
        "Confidence": (
            f"{result['confidence']:.2f}%"
            if result["confidence"] is not None
            else "Unavailable"
        ),
        "Status": result["status"],
        "Top Predictions": get_top_predictions(
            model,
            reading,
            top_n=3
        ) if hasattr(model, "predict_proba") else []
    }

    return summary
