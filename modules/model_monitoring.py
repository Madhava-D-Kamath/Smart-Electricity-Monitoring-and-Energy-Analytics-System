
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "model_monitoring"
HISTORY_PATH = RESULTS_DIR / "model_performance_history.csv"


def validate_evaluation_data(X_test, y_test):
    """Validate test data before model evaluation."""

    if X_test is None or y_test is None:
        raise ValueError("Test features and labels are required.")

    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("Test dataset cannot be empty.")

    if len(X_test) != len(y_test):
        raise ValueError(
            "Test features and labels must have equal lengths."
        )

    if pd.Series(y_test).isna().any():
        raise ValueError("Test labels cannot contain missing values.")


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str = "Appliance Classifier",
    save_history: bool = True
) -> dict[str, Any]:
    """
    Evaluate a trained classification model.

    Returns metrics, confusion matrix, classification report,
    and actual-versus-predicted results.
    """

    validate_evaluation_data(X_test, y_test)

    if model is None or not hasattr(model, "predict"):
        raise ValueError("A trained model with predict() is required.")

    predictions = model.predict(X_test)

    if len(predictions) != len(y_test):
        raise ValueError(
            "Model predictions and test labels have different lengths."
        )

    actual = pd.Series(y_test).reset_index(drop=True)
    predicted = pd.Series(predictions).reset_index(drop=True)

    labels = sorted(
        set(actual.tolist()) | set(predicted.tolist()),
        key=str
    )

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual,
        predicted,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        average="weighted",
        zero_division=0
    )

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=labels
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=[f"Actual: {label}" for label in labels],
        columns=[f"Predicted: {label}" for label in labels]
    )

    report = classification_report(
        actual,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    comparison = pd.DataFrame({
        "actual_appliance": actual,
        "predicted_appliance": predicted
    })

    comparison["correct_prediction"] = (
        comparison["actual_appliance"]
        == comparison["predicted_appliance"]
    )

    metrics = {
        "model_name": model_name,
        "evaluation_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "total_test_samples": int(len(actual)),
        "correct_predictions": int(
            comparison["correct_prediction"].sum()
        ),
        "incorrect_predictions": int(
            (~comparison["correct_prediction"]).sum()
        )
    }

    if save_history:
        save_model_performance(metrics)

    return {
        "metrics": metrics,
        "confusion_matrix": confusion_df,
        "classification_report": report,
        "prediction_comparison": comparison
    }


def save_model_performance(
    metrics: dict,
    file_path: Path = HISTORY_PATH
) -> Path:
    """Append model evaluation metrics to a CSV history file."""

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    new_record = pd.DataFrame([metrics])

    if file_path.exists():
        existing = pd.read_csv(file_path)

        updated = pd.concat(
            [existing, new_record],
            ignore_index=True
        )
    else:
        updated = new_record

    updated.to_csv(file_path, index=False)

    return file_path


def load_model_performance_history(
    file_path: Path = HISTORY_PATH
) -> pd.DataFrame:
    """Load previously recorded model evaluation results."""

    file_path = Path(file_path)

    if not file_path.exists():
        return pd.DataFrame(
            columns=[
                "model_name",
                "evaluation_timestamp",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "total_test_samples",
                "correct_predictions",
                "incorrect_predictions"
            ]
        )

    return pd.read_csv(file_path)


def compare_model_performance(
    history: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Compare models using their recorded evaluation metrics."""

    if history is None:
        history = load_model_performance_history()

    if history.empty:
        return pd.DataFrame()

    required = [
        "model_name",
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]

    missing = [
        column for column in required
        if column not in history.columns
    ]

    if missing:
        raise ValueError(
            f"Performance history is missing columns: {missing}"
        )

    return history.sort_values(
        by="f1_score",
        ascending=False
    ).reset_index(drop=True)


def get_best_model(
    history: pd.DataFrame | None = None
) -> dict | None:
    """Return the best recorded model based on weighted F1-score."""

    comparison = compare_model_performance(history)

    if comparison.empty:
        return None

    best = comparison.iloc[0]

    return best.to_dict()


def generate_monitoring_report(
    model,
    X_test,
    y_test,
    model_name: str = "Appliance Classifier"
) -> dict:
    """Generate an evaluation and monitoring report."""

    evaluation = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name
    )

    history = load_model_performance_history()

    return {
        "evaluation": evaluation,
        "performance_history": history,
        "best_recorded_model": get_best_model(history)
    }
