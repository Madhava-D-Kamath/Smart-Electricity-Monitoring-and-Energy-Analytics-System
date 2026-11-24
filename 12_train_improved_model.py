import os
import warnings
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

warnings.filterwarnings("ignore")


# ==========================================================
# CONFIGURATION
# ==========================================================

ORIGINAL_DATA_PATH = "results/ml_ready_dataset_active.csv"

NEW_DATASETS = {
    "television": "dataset/new_appliances/television/television.csv",
    "refrigerator": "dataset/new_appliances/refrigerator/refrigerator.csv",
    "mixer grinder": "dataset/new_appliances/mixer_grinder/mixer_grinder.csv",
    "washing machine": "dataset/new_appliances/washing_machine/washing_machine.csv"
}

MODEL_OUTPUT_PATH = "models/random_forest_10_appliances_candidate.pkl"

RESULTS_OUTPUT_PATH = "results/10_appliance_model_comparison.csv"

FEATURES = [
    "voltage_V",
    "current_A",
    "power_W",
    "frequency_Hz",
    "power_factor"
]

TARGET = "appliance"

REQUIRED_COLUMNS = [
    "timestamp",
    "session_id",
    "appliance",
    "voltage_V",
    "current_A",
    "power_W",
    "frequency_Hz",
    "power_factor"
]


# ==========================================================
# LOAD ORIGINAL DATASET
# ==========================================================

print("\n==========================================")
print("LOADING ORIGINAL DATASET")
print("==========================================")

if not os.path.exists(ORIGINAL_DATA_PATH):
    raise FileNotFoundError(
        f"Original dataset not found: {ORIGINAL_DATA_PATH}"
    )

original_df = pd.read_csv(ORIGINAL_DATA_PATH)

missing_columns = [
    col for col in FEATURES + [TARGET, "session_id"]
    if col not in original_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in original dataset: {missing_columns}"
    )

original_df["source"] = "original"

print("Original records:", len(original_df))

print("\nOriginal appliance distribution:")
print(original_df[TARGET].value_counts())


# ==========================================================
# LOAD FOUR NEW APPLIANCE DATASETS
# ==========================================================

print("\n==========================================")
print("LOADING NEW APPLIANCE DATASETS")
print("==========================================")

new_dataframes = []

for appliance_name, file_path in NEW_DATASETS.items():

    print(f"\nChecking: {appliance_name}")

    if not os.path.exists(file_path):

        print("WARNING: CSV file not found. Skipping.")
        continue

    new_df = pd.read_csv(file_path)

    if new_df.empty:

        print("CSV is empty. Waiting for sensor readings.")
        continue

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in new_df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{file_path} is missing columns: {missing_columns}"
        )

    # Normalize the appliance label.
    new_df["appliance"] = (
        new_df["appliance"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Use the folder's appliance name for blank labels.
    blank_labels = new_df["appliance"] == ""

    new_df.loc[blank_labels, "appliance"] = appliance_name

    # Reject incorrectly labelled rows.
    invalid_labels = (
        new_df["appliance"].str.lower() != appliance_name.lower()
    )

    if invalid_labels.any():

        raise ValueError(
            f"Unexpected appliance labels in {file_path}. "
            f"Expected '{appliance_name}'. "
            f"Found: {new_df.loc[invalid_labels, 'appliance'].unique()}"
        )

    new_df["appliance"] = appliance_name

    new_df["source"] = "new"

    new_dataframes.append(new_df)

    print("Records loaded:", len(new_df))
    print("Appliance:", appliance_name)


# ==========================================================
# COMBINE ALL DATASETS
# ==========================================================

print("\n==========================================")
print("COMBINING DATASETS")
print("==========================================")

all_dataframes = [original_df] + new_dataframes

df = pd.concat(
    all_dataframes,
    ignore_index=True,
    sort=False
)

print("Total combined records:", len(df))

print("\nCombined appliance distribution:")
print(df["appliance"].value_counts())


# ==========================================================
# DATA CLEANING
# ==========================================================

print("\n==========================================")
print("DATA CLEANING")
print("==========================================")

# Remove leading/trailing spaces from labels.
df["appliance"] = df["appliance"].astype(str).str.strip()

# Convert model features into numeric values.
for column in FEATURES:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Remove records with missing model features or labels.
before_cleaning = len(df)

df = df.dropna(
    subset=FEATURES + [TARGET, "session_id"]
).copy()

# Remove rows with infinite numeric values.
df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=FEATURES
).copy()

# Remove rows with blank session IDs.
df["session_id"] = df["session_id"].astype(str).str.strip()

df = df[
    (df["session_id"] != "") &
    (df["session_id"].str.lower() != "nan") &
    (df["appliance"] != "")
].copy()

# Physical measurement sanity checks.
# These are broad validity checks, not appliance-specific limits.
df = df[
    (df["voltage_V"] > 0) &
    (df["current_A"] >= 0) &
    (df["power_W"] >= 0) &
    (df["frequency_Hz"] > 0) &
    (df["power_factor"].between(0, 1))
].copy()

print("Records before cleaning:", before_cleaning)
print("Records after cleaning:", len(df))
print("Records removed:", before_cleaning - len(df))


# ==========================================================
# APPLIANCE DISTRIBUTION
# ==========================================================

print("\n==========================================")
print("FINAL APPLIANCE DISTRIBUTION")
print("==========================================")

print(df["appliance"].value_counts())

print("\nNumber of appliance classes:", df["appliance"].nunique())


# ==========================================================
# SESSION INFORMATION
# ==========================================================

print("\n==========================================")
print("SESSION INFORMATION")
print("==========================================")

# Composite group ID prevents sessions from different
# appliance types being treated as the same session.

df["group_id"] = (
    df["appliance"].astype(str) +
    "::" +
    df["session_id"].astype(str)
)

session_counts = (
    df.groupby("appliance")["group_id"]
    .nunique()
    .sort_values()
)

print("\nIndependent sessions per appliance:")
print(session_counts)

# A class needs at least two independent sessions
# to appear in both training and testing.

insufficient_sessions = session_counts[session_counts < 2]

if not insufficient_sessions.empty:

    raise ValueError(
        "\nSome appliance classes have fewer than two independent "
        "sessions:\n"
        f"{insufficient_sessions}\n\n"
        "Collect more independent sessions for these appliances "
        "before training with session-based validation."
    )


# ==========================================================
# SELECT FEATURES AND TARGET
# ==========================================================

X = df[FEATURES].copy()

y = df[TARGET].copy()

groups = df["group_id"].copy()

all_classes = sorted(y.unique())

print("\nFeatures:")
print(FEATURES)

print("\nClasses:")
print(all_classes)


# ==========================================================
# SESSION-BASED TRAIN / TEST SPLIT
# ==========================================================

print("\n==========================================")
print("SESSION-BASED TRAIN / TEST SPLIT")
print("==========================================")

# Search for a group-based split that contains every
# appliance class in both training and testing.

split_found = False

for random_state in range(42, 2042):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=random_state
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups)
    )

    y_train_candidate = y.iloc[train_indices]
    y_test_candidate = y.iloc[test_indices]

    if (
        set(all_classes).issubset(set(y_train_candidate.unique()))
        and
        set(all_classes).issubset(set(y_test_candidate.unique()))
    ):

        split_found = True
        break


if not split_found:

    raise ValueError(
        "Could not find a session-based split containing every "
        "appliance class in both training and testing.\n"
        "Collect more independent sessions per appliance."
    )


X_train = X.iloc[train_indices].copy()
X_test = X.iloc[test_indices].copy()

y_train = y.iloc[train_indices].copy()
y_test = y.iloc[test_indices].copy()

print("Random state selected:", random_state)

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


# ==========================================================
# MODEL 1: STANDARD RANDOM FOREST
# ==========================================================

print("\n==========================================")
print("TRAINING STANDARD RANDOM FOREST")
print("==========================================")

standard_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

standard_model.fit(
    X_train,
    y_train
)

standard_predictions = standard_model.predict(X_test)

print("Standard Random Forest training completed.")


# ==========================================================
# MODEL 2: BALANCED RANDOM FOREST
# ==========================================================

print("\n==========================================")
print("TRAINING BALANCED RANDOM FOREST")
print("==========================================")

balanced_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

balanced_model.fit(
    X_train,
    y_train
)

balanced_predictions = balanced_model.predict(X_test)

print("Balanced Random Forest training completed.")


# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate_model(name, model, predictions):

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\n==========================================")
    print(name)
    print("==========================================")

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(f"Macro F1: {macro_f1:.4f}")

    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=all_classes,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=all_classes
    )

    print(
        pd.DataFrame(
            matrix,
            index=all_classes,
            columns=all_classes
        )
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1
    }


# ==========================================================
# EVALUATE BOTH MODELS
# ==========================================================

standard_results = evaluate_model(
    "Standard Random Forest",
    standard_model,
    standard_predictions
)

balanced_results = evaluate_model(
    "Balanced Random Forest",
    balanced_model,
    balanced_predictions
)


# ==========================================================
# MODEL COMPARISON
# ==========================================================

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

comparison_df = pd.DataFrame(
    [standard_results, balanced_results]
)

comparison_df = comparison_df.sort_values(
    by=["Macro F1", "Balanced Accuracy"],
    ascending=False
)

print(
    comparison_df.to_string(index=False)
)

os.makedirs("results", exist_ok=True)

comparison_df.to_csv(
    RESULTS_OUTPUT_PATH,
    index=False
)

print("\nComparison saved:")
print(RESULTS_OUTPUT_PATH)


# ==========================================================
# SELECT BEST MODEL
# ==========================================================

print("\n==========================================")
print("SELECTING BEST MODEL")
print("==========================================")

if (
    balanced_results["Macro F1"],
    balanced_results["Balanced Accuracy"]
) > (
    standard_results["Macro F1"],
    standard_results["Balanced Accuracy"]
):

    best_model = balanced_model
    best_name = "Balanced Random Forest"
    best_results = balanced_results

else:

    best_model = standard_model
    best_name = "Standard Random Forest"
    best_results = standard_results


# ==========================================================
# SAVE NEW CANDIDATE MODEL
# ==========================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    MODEL_OUTPUT_PATH
)

print("\n==========================================")
print("TRAINING COMPLETED")
print("==========================================")

print("Selected model:", best_name)

print(
    "Selected Macro F1:",
    round(best_results["Macro F1"], 4)
)

print(
    "Selected Balanced Accuracy:",
    round(best_results["Balanced Accuracy"], 4)
)

print("\nModel saved successfully:")
print(MODEL_OUTPUT_PATH)

print("\nComparison file:")
print(RESULTS_OUTPUT_PATH)

print("\nAppliance classes in trained model:")
print(list(best_model.classes_))

print("\nExisting models were not overwritten.")