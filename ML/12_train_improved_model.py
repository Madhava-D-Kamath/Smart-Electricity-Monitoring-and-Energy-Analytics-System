import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# ==========================================
# LOAD CLEANED DATASET
# ==========================================

DATA_PATH = "results/ml_ready_dataset_active.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET INFORMATION ==========")

print("Total records:", len(df))

print("\nAppliance distribution:")
print(df["appliance"].value_counts())

# ==========================================
# SELECT EXACT MODEL FEATURES
# ==========================================

FEATURES = [
    "voltage_V",
    "current_A",
    "power_W",
    "frequency_Hz",
    "power_factor"
]

TARGET = "appliance"

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = df[TARGET]

valid = X.notna().all(axis=1) & y.notna()

X = X.loc[valid]
y = y.loc[valid]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# MODEL 1: STANDARD RANDOM FOREST
# ==========================================

standard_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

standard_model.fit(X_train, y_train)

standard_predictions = standard_model.predict(X_test)

# ==========================================
# MODEL 2: BALANCED RANDOM FOREST
# ==========================================

balanced_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

balanced_model.fit(X_train, y_train)

balanced_predictions = balanced_model.predict(X_test)

# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, model, predictions):

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    print("\n========================================")
    print(name)
    print("========================================")

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=model.classes_
        )
    )

    return accuracy, macro_f1, weighted_f1


# ==========================================
# COMPARE BOTH MODELS
# ==========================================

standard_results = evaluate_model(
    "STANDARD RANDOM FOREST",
    standard_model,
    standard_predictions
)

balanced_results = evaluate_model(
    "BALANCED RANDOM FOREST",
    balanced_model,
    balanced_predictions
)

# ==========================================
# SAVE CANDIDATE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

# Select based on macro F1
if balanced_results[1] > standard_results[1]:

    best_model = balanced_model
    best_name = "Balanced Random Forest"

else:

    best_model = standard_model
    best_name = "Standard Random Forest"

joblib.dump(
    best_model,
    "models/random_forest_appliance_candidate.pkl"
)

print("\n========================================")
print("TRAINING COMPLETED")
print("========================================")

print("Selected model:", best_name)

print(
    "Candidate model saved:",
    "models/random_forest_appliance_candidate.pkl"
)

print("\nOriginal model has NOT been modified.")