"""
Train and compare appliance-classification models.
Run from the repository root:
    python scripts/train_compare_models.py

Models are evaluated on the same stratified hold-out split. If your dataset
contains repeated readings from the same appliance session, use a
session/group-based split for a stronger leakage-resistant evaluation.
"""
from pathlib import Path
import sys
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, classification_report
)

ROOT = Path(__file__).resolve().parents[1]

# Add project root to Python import path
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "results" / "ml_ready_dataset.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
FEATURES = ["voltage_V", "current_A", "power_W", "frequency_Hz", "power_factor"]
TARGET = "appliance"
RANDOM_STATE = 42

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(str).str.strip()
    valid = y.ne("") & y.notna() & X.notna().any(axis=1)
    X, y = X.loc[valid], y.loc[valid]
    if y.nunique() < 2:
        raise ValueError("At least two appliance classes are required.")
    counts = y.value_counts()
    if counts.min() < 2:
        raise ValueError("Each appliance class needs at least 2 records for stratified splitting.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=18, min_samples_leaf=1,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, random_state=RANDOM_STATE
        ),
    }

    try:
        from ml_models import XGBoostApplianceClassifier
        candidates["XGBoost"] = XGBoostApplianceClassifier(
            n_estimators=350, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1
        )
    except ImportError:
        print("XGBoost not installed; skipping it. Install with: pip install xgboost")

    MODEL_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    rows, reports, manifest = [], {}, {}

    for display_name, estimator in candidates.items():
        safe_name = (
            display_name.lower().replace(" ", "_")
            .replace("(", "").replace(")", "")
        )
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", estimator)
        ])
        print(f"\nTraining {display_name} ...")
        try:
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            metrics = {
                "Model": display_name,
                "Accuracy": accuracy_score(y_test, pred),
                "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
                "Precision (Macro)": precision_score(y_test, pred, average="macro", zero_division=0),
                "Recall (Macro)": recall_score(y_test, pred, average="macro", zero_division=0),
                "F1 (Macro)": f1_score(y_test, pred, average="macro", zero_division=0),
                "F1 (Weighted)": f1_score(y_test, pred, average="weighted", zero_division=0),
            }
            model_file = f"{safe_name}.joblib"
            joblib.dump(pipeline, MODEL_DIR / model_file)
            rows.append(metrics)
            reports[display_name] = classification_report(
                y_test, pred, zero_division=0, output_dict=True
            )
            manifest[display_name] = model_file
            print(f"Accuracy: {metrics['Accuracy']*100:.2f}% | Macro F1: {metrics['F1 (Macro)']:.4f}")
        except Exception as exc:
            warnings.warn(f"{display_name} failed: {exc}")

    if not rows:
        raise RuntimeError("No model trained successfully.")

    results = pd.DataFrame(rows).sort_values(
        ["F1 (Macro)", "Balanced Accuracy", "Accuracy"], ascending=False
    )
    results.to_csv(RESULT_DIR / "model_comparison.csv", index=False)
    with open(RESULT_DIR / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    with open(MODEL_DIR / "model_registry.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== MODEL COMPARISON (ranked by Macro F1) ===")
    print(results.to_string(index=False, formatters={
        c: (lambda x: f"{x*100:.2f}%") for c in
        ["Accuracy", "Balanced Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)", "F1 (Weighted)"]
    }))
    print(f"\nSaved model files in: {MODEL_DIR}")
    print(f"Comparison CSV: {RESULT_DIR / 'model_comparison.csv'}")
    print("Choose the best model using Macro F1 and Balanced Accuracy, not accuracy alone.")

if __name__ == "__main__":
    main()
