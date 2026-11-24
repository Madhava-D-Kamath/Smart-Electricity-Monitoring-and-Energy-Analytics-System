import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================
# 1. LOAD UPDATED DATASET
# =====================================

df = pd.read_csv("results/ml_ready_dataset.csv")

print("\n========== RANDOM FOREST TRAINING ==========")

print("\nTotal ML records:", len(df))

# =====================================
# 2. CHECK DATASET
# =====================================

print("\nAppliance distribution:")
print(df["appliance"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

# =====================================
# 3. SELECT FEATURES
# =====================================

features = [
    "voltage_V",
    "current_A",
    "power_W",
    "frequency_Hz",
    "power_factor"
]

X = df[features]

y = df["appliance"]

# =====================================
# 4. SPLIT DATA
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# =====================================
# 5. CREATE RANDOM FOREST MODEL
# =====================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# =====================================
# 6. TRAIN MODEL
# =====================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")

# =====================================
# 7. MAKE PREDICTIONS
# =====================================

y_pred = model.predict(X_test)

# =====================================
# 8. EVALUATE MODEL
# =====================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))

# =====================================
# 9. FEATURE IMPORTANCE
# =====================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# =====================================
# 10. SAVE MODEL
# =====================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

joblib.dump(
    model,
    "models/random_forest_appliance.pkl"
)

importance.to_csv(
    "results/feature_importance.csv",
    index=False
)

print("\nModel saved successfully!")

print("Location: models/random_forest_appliance.pkl")

print("\n========== TRAINING COMPLETED ==========")
