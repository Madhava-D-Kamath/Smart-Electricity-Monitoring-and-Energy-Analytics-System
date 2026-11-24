
"""
==========================================================
AI SMART ELECTRICITY MONITORING SYSTEM
MODULE: ANOMALY DETECTION
==========================================================

Features:
1. Detect unusual electrical measurements.
2. Identify abnormal power consumption.
3. Detect unusual current and voltage patterns.
4. Train an Isolation Forest model.
5. Generate anomaly severity reports.
6. Support single and batch predictions.

Algorithm:
Isolation Forest
==========================================================
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

FEATURES = [
    "voltage_V",
    "current_A",
    "power_W",
    "frequency_Hz",
    "power_factor"
]

MODEL_PATH = "models/anomaly_detection_model.pkl"


# ==========================================================
# 2. DATA VALIDATION
# ==========================================================

def validate_data(df):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    data = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if data.isna().any().any():
        raise ValueError(
            "Dataset contains missing or invalid measurements."
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
# 3. TRAIN ISOLATION FOREST
# ==========================================================

def train_anomaly_model(
    df,
    contamination=0.02,
    random_state=42
):
    """
    Train an Isolation Forest model.

    Important:
    Train on representative normal operating readings.
    Do not assume every historical reading is normal.
    """

    if not 0 < contamination < 0.5:
        raise ValueError(
            "Contamination must be between 0 and 0.5."
        )

    data = validate_data(df)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(data)

    return model


# ==========================================================
# 4. SAVE MODEL
# ==========================================================

def save_anomaly_model(
    model,
    model_path=MODEL_PATH
):

    directory = os.path.dirname(model_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    joblib.dump(
        model,
        model_path
    )

    return model_path


# ==========================================================
# 5. LOAD MODEL
# ==========================================================

def load_anomaly_model(
    model_path=MODEL_PATH
):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Anomaly model not found: {model_path}"
        )

    return joblib.load(model_path)


# ==========================================================
# 6. DETECT ANOMALIES
# ==========================================================

def detect_anomalies(
    model,
    df
):
    """
    Detect unusual readings.

    Output:
        1  = Normal
       -1  = Anomaly
    """

    data = validate_data(df)

    predictions = model.predict(data)

    scores = model.decision_function(data)

    results = df.copy()

    results["anomaly_prediction"] = predictions

    results["anomaly_status"] = np.where(
        predictions == -1,
        "Anomaly",
        "Normal"
    )

    results["anomaly_score"] = scores

    return results


# ==========================================================
# 7. SINGLE READING ANOMALY DETECTION
# ==========================================================

def detect_single_anomaly(
    model,
    reading
):
    """
    Analyse one electrical reading.
    """

    input_df = pd.DataFrame(
        [reading]
    )

    results = detect_anomalies(
        model,
        input_df
    )

    row = results.iloc[0]

    return {
        "status": row["anomaly_status"],
        "prediction": int(row["anomaly_prediction"]),
        "anomaly_score": float(row["anomaly_score"])
    }


# ==========================================================
# 8. ANOMALY SUMMARY
# ==========================================================

def get_anomaly_summary(results):
    """
    Generate anomaly statistics.
    """

    if "anomaly_status" not in results.columns:
        raise ValueError(
            "Run detect_anomalies() before generating the summary."
        )

    total = len(results)

    anomalies = int(
        (results["anomaly_status"] == "Anomaly").sum()
    )

    normal = total - anomalies

    anomaly_percentage = (
        anomalies / total * 100
        if total > 0 else 0
    )

    return {
        "total_readings": total,
        "normal_readings": normal,
        "anomalies": anomalies,
        "anomaly_percentage": round(
            anomaly_percentage,
            2
        )
    }
