"""
SESSION-BASED VALIDATION OF APPLIANCE CLASSIFICATION MODELS

Purpose:
1. Load the existing ML-ready dataset.
2. Validate appliance labels and session IDs.
3. Create a leakage-resistant session-based train/test split.
4. Train seven classification models.
5. Evaluate all models on the same unseen sessions.
6. Save comparison metrics, classification reports,
   confusion matrices and trained candidate models.

IMPORTANT:
- Existing models are not overwritten.
- Existing evaluation files are not overwritten.
- Every appliance must have at least two independent sessions.
"""

from pathlib import Path
import sys
import json
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GroupShuffleSplit

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================================
# CONFIGURATION
# ==========================================================

warnings.filterwarnings("default")

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "results" / "ml_ready_dataset.csv"

OUTPUT_DIR = ROOT / "results" / "session_validation"

MODEL_OUTPUT_DIR = ROOT / "models" / "session_validated"

FEATURES = [
    "voltage_V",
    "current_A",
    "power_W",
    "frequency_Hz",
    "power_factor"
]

TARGET = "appliance"

SESSION_COLUMN = "session_id"

RANDOM_STATE = 42

TEST_SIZE = 0.20

MAX_SPLIT_ATTEMPTS = 2000


# ==========================================================
# CREATE OUTPUT DIRECTORIES
# ==========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# LOAD DATASET
# ==========================================================

print("\n" + "=" * 65)
print("SESSION-BASED MODEL VALIDATION")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")

print("Dataset:", DATA_PATH.name)

print("Total records:", len(df))


# ==========================================================
# VALIDATE REQUIRED COLUMNS
# ==========================================================

required_columns = FEATURES + [
    TARGET,
    SESSION_COLUMN
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ==========================================================
# DATA CLEANING
# ==========================================================

print("\n" + "=" * 65)
print("DATA CLEANING")
print("=" * 65)

df = df.copy()

# Clean appliance labels.

df[TARGET] = (
    df[TARGET]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Clean session IDs.

df[SESSION_COLUMN] = (
    df[SESSION_COLUMN]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Convert model features into numeric values.

for column in FEATURES:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Replace infinite values.

df[FEATURES] = df[FEATURES].replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove invalid labels and session IDs.

df = df[
    (df[TARGET] != "") &
    (df[SESSION_COLUMN] != "") &
    (df[SESSION_COLUMN].str.lower() != "nan")
].copy()

# Remove rows where every model feature is missing.

df = df[
    df[FEATURES].notna().any(axis=1)
].copy()

if df.empty:
    raise ValueError(
        "No valid records remain after cleaning."
    )

print("Records after cleaning:", len(df))

print("\nAppliance distribution:")

print(df[TARGET].value_counts())


# ==========================================================
# CREATE COMPOSITE SESSION GROUP
# ==========================================================

print("\n" + "=" * 65)
print("SESSION GROUP INFORMATION")
print("=" * 65)

# Different appliances may reuse the same session ID.
# The composite identifier keeps their groups separate.

df["group_id"] = (
    df[TARGET].astype(str)
    + "::"
    + df[SESSION_COLUMN].astype(str)
)

session_counts = (
    df.groupby(TARGET)["group_id"]
    .nunique()
    .sort_values()
)

print("\nIndependent sessions per appliance:")

print(session_counts.to_string())

session_counts.to_csv(
    OUTPUT_DIR / "session_counts.csv",
    header=["independent_sessions"]
)


# ==========================================================
# CHECK MINIMUM NUMBER OF INDEPENDENT SESSIONS
# ==========================================================

insufficient_sessions = session_counts[
    session_counts < 2
]

if not insufficient_sessions.empty:

    print("\nVALIDATION CANNOT CONTINUE.")

    print(
        "\nThe following appliance classes have fewer than "
        "two independent sessions:"
    )

    print(insufficient_sessions.to_string())

    print(
        "\nCollect additional independent sessions for "
        "these appliances before session-based validation."
    )

    raise ValueError(
        "Insufficient independent sessions. "
        "No model training was performed."
    )


# ==========================================================
# PREPARE FEATURES AND TARGET
# ==========================================================

X = df[FEATURES].copy()

y = df[TARGET].copy()

groups = df["group_id"].copy()

all_classes = sorted(y.unique())

print("\nFeatures:")

print(FEATURES)

print("\nAppliance classes:")

print(all_classes)


# ==========================================================
# SESSION-BASED TRAIN / TEST SPLIT
# ==========================================================

print("\n" + "=" * 65)
print("SESSION-BASED TRAIN / TEST SPLIT")
print("=" * 65)

split_found = False

selected_random_state = None

for random_state in range(
    RANDOM_STATE,
    RANDOM_STATE + MAX_SPLIT_ATTEMPTS
):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=random_state
    )

    train_indices, test_indices = next(
        splitter.split(
            X,
            y,
            groups
        )
    )

    y_train_candidate = y.iloc[train_indices]

    y_test_candidate = y.iloc[test_indices]

    train_classes = set(
        y_train_candidate.unique()
    )

    test_classes = set(
        y_test_candidate.unique()
    )

    # Every appliance must appear in both datasets.

    if not set(all_classes).issubset(train_classes):
        continue

    if not set(all_classes).issubset(test_classes):
        continue

    # Verify that no session appears in both sets.

    train_groups = set(
        groups.iloc[train_indices]
    )

    test_groups = set(
        groups.iloc[test_indices]
    )

    if train_groups.intersection(test_groups):
        continue

    split_found = True

    selected_random_state = random_state

    break


if not split_found:

    raise ValueError(
        "Could not find a valid session-based split "
        "containing every appliance class in both sets.\n"
        "Collect additional independent sessions or "
        "review the session distribution."
    )


# ==========================================================
# CREATE FINAL TRAINING AND TESTING DATA
# ==========================================================

X_train = X.iloc[train_indices].copy()

X_test = X.iloc[test_indices].copy()

y_train = y.iloc[train_indices].copy()

y_test = y.iloc[test_indices].copy()

groups_train = groups.iloc[train_indices]

groups_test = groups.iloc[test_indices]


print("\nSelected random state:", selected_random_state)

print("Training records:", len(X_train))

print("Testing records:", len(X_test))

print(
    "Training percentage:",
    round(len(X_train) / len(df) * 100, 2),
    "%"
)

print(
    "Testing percentage:",
    round(len(X_test) / len(df) * 100, 2),
    "%"
)

print("\nTraining appliance distribution:")

print(y_train.value_counts())

print("\nTesting appliance distribution:")

print(y_test.value_counts())

print("\nTraining independent sessions:")

print(groups_train.nunique())

print("\nTesting independent sessions:")

print(groups_test.nunique())

print("\nSession overlap:")

print(
    len(
        set(groups_train).intersection(
            set(groups_test)
        )
    )
)

# Save the exact split assignment for reproducibility.

split_assignment = df[
    [TARGET, SESSION_COLUMN, "group_id"]
].copy()

split_assignment["split"] = np.where(
    df.index.isin(X_train.index),
    "train",
    "test"
)

split_assignment.to_csv(
    OUTPUT_DIR / "session_split_assignment.csv",
    index=False
)


# ==========================================================
# DEFINE CLASSIFICATION MODELS
# ==========================================================

print("\n" + "=" * 65)
print("INITIALIZING CLASSIFICATION MODELS")
print("=" * 65)

candidates = {

    "Baseline (Most Frequent)": DummyClassifier(
        strategy="most_frequent"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),

    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        random_state=RANDOM_STATE
    )
}


# ==========================================================
# ADD XGBOOST
# ==========================================================

try:

    from ml_models import XGBoostApplianceClassifier

    candidates["XGBoost"] = XGBoostApplianceClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("XGBoost available.")

except ImportError:

    print(
        "XGBoost unavailable. "
        "The other six models will be evaluated."
    )


# ==========================================================
# MODEL EVALUATION FUNCTION
# ==========================================================

def evaluate_model(
    display_name,
    pipeline
):

    print("\n" + "=" * 65)

    print("EVALUATING:", display_name)

    print("=" * 65)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    precision_macro = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(
        f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Precision Macro: "
        f"{precision_macro * 100:.2f}%"
    )

    print(
        f"Recall Macro: "
        f"{recall_macro * 100:.2f}%"
    )

    print(
        f"F1 Macro: {f1_macro * 100:.2f}%"
    )

    print(
        f"F1 Weighted: {f1_weighted * 100:.2f}%"
    )

    # Classification report.

    report = classification_report(
        y_test,
        predictions,
        labels=all_classes,
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix.

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=all_classes
    )

    # Create safe model filename.

    safe_name = (
        display_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    # Save classification report.

    with open(
        OUTPUT_DIR / f"{safe_name}_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    # Save confusion matrix CSV.

    matrix_df = pd.DataFrame(
        matrix,
        index=all_classes,
        columns=all_classes
    )

    matrix_df.to_csv(
        OUTPUT_DIR / f"{safe_name}_confusion_matrix.csv"
    )

    # Save confusion matrix image.

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=all_classes
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False
    )

    ax.set_title(
        f"{display_name} - Session-Based Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{safe_name}_confusion_matrix.png",
        dpi=200
    )

    plt.close(fig)

    # Save the newly trained model separately.

    model_path = (
        MODEL_OUTPUT_DIR / f"{safe_name}.joblib"
    )

    joblib.dump(
        pipeline,
        model_path
    )

    return {

        "Model": display_name,

        "Accuracy": accuracy,

        "Balanced Accuracy": balanced_accuracy,

        "Precision (Macro)": precision_macro,

        "Recall (Macro)": recall_macro,

        "F1 (Macro)": f1_macro,

        "F1 (Weighted)": f1_weighted,

        "Test Samples": len(y_test),

        "Model File": str(model_path.relative_to(ROOT))
    }


# ==========================================================
# TRAIN AND EVALUATE ALL MODELS
# ==========================================================

print("\n" + "=" * 65)

print("TRAINING AND EVALUATING MODELS")

print("=" * 65)

results = []

reports = {}

failed_models = []

for display_name, estimator in candidates.items():

    print("\nTraining:", display_name)

    pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "classifier",
            estimator
        )

    ])

    try:

        pipeline.fit(
            X_train,
            y_train
        )

        model_result = evaluate_model(
            display_name,
            pipeline
        )

        results.append(model_result)

        print(
            "\nCompleted:",
            display_name
        )

    except Exception as error:

        failed_models.append({

            "Model": display_name,

            "Error": str(error)

        })

        warnings.warn(
            f"{display_name} failed: {error}"
        )


# ==========================================================
# SAVE MODEL COMPARISON
# ==========================================================

if not results:

    raise RuntimeError(
        "No models were successfully evaluated."
    )


comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by=[
        "F1 (Macro)",
        "Balanced Accuracy",
        "Accuracy"
    ],
    ascending=False
)

comparison_df.to_csv(
    OUTPUT_DIR / "session_model_comparison.csv",
    index=False
)


# ==========================================================
# SAVE MODEL REGISTRY
# ==========================================================

session_registry = {

    row["Model"]: row["Model File"]

    for row in results

}

with open(
    OUTPUT_DIR / "session_model_registry.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        session_registry,
        file,
        indent=2
    )


# ==========================================================
# SAVE VALIDATION METADATA
# ==========================================================

metadata = {

    "validation_type": "GroupShuffleSplit",

    "dataset": str(DATA_PATH.relative_to(ROOT)),

    "features": FEATURES,

    "target": TARGET,

    "test_size": TEST_SIZE,

    "random_state": selected_random_state,

    "training_records": len(X_train),

    "testing_records": len(X_test),

    "total_records": len(df),

    "training_sessions": int(groups_train.nunique()),

    "testing_sessions": int(groups_test.nunique()),

    "session_overlap": 0,

    "appliance_classes": all_classes,

    "successful_models": len(results),

    "failed_models": failed_models

}

with open(
    OUTPUT_DIR / "validation_metadata.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=2
    )


# ==========================================================
# FINAL RESULTS
# ==========================================================

print("\n" + "=" * 65)

print("SESSION-BASED MODEL COMPARISON")

print("=" * 65)

print(
    comparison_df.to_string(
        index=False,
        formatters={

            column: lambda value: f"{value * 100:.2f}%"

            for column in [

                "Accuracy",

                "Balanced Accuracy",

                "Precision (Macro)",

                "Recall (Macro)",

                "F1 (Macro)",

                "F1 (Weighted)"

            ]

        }
    )
)


# ==========================================================
# BEST MODEL
# ==========================================================

best_model = comparison_df.iloc[0]

print("\n" + "=" * 65)

print("BEST SESSION-VALIDATED MODEL")

print("=" * 65)

print("Model:", best_model["Model"])

print(
    "Accuracy:",
    f"{best_model['Accuracy'] * 100:.2f}%"
)

print(
    "Macro F1:",
    f"{best_model['F1 (Macro)'] * 100:.2f}%"
)

print(
    "Balanced Accuracy:",
    f"{best_model['Balanced Accuracy'] * 100:.2f}%"
)


# ==========================================================
# COMPLETION
# ==========================================================

print("\n" + "=" * 65)

print("SESSION VALIDATION COMPLETED")

print("=" * 65)

print("\nResults saved in:")

print(OUTPUT_DIR)

print("\nNew candidate models saved in:")

print(MODEL_OUTPUT_DIR)

print("\nExisting models were not overwritten.")

print("\nExisting evaluation files were not overwritten.")