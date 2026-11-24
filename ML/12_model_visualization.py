import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

print("\n========== MODEL VISUALIZATION ==========")

# Load dataset
df = pd.read_csv("results/ml_ready_dataset.csv")

# Exclude uncertain Session 7
df = df[df["session_id"] != 7].copy()

features = [
    "voltage_V",
    "current_A",
    "power_W",
    "frequency_Hz",
    "power_factor"
]

X = df[features]
y = df["appliance"]

# Reproduce the same train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Load trained model
model = joblib.load("models/random_forest_appliance.pkl")

# Predictions
y_pred = model.predict(X_test)

os.makedirs("results", exist_ok=True)

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Appliance")
plt.ylabel("Actual Appliance")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.show()

# 2. Feature Importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(importance["Feature"], importance["Importance"])
plt.title("Random Forest - Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Electrical Features")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=300)
plt.show()

# 3. Classification Performance
report = classification_report(
    y_test, y_pred, output_dict=True, zero_division=0
)

performance = pd.DataFrame(report).T
performance = performance.loc[
    model.classes_, ["precision", "recall", "f1-score"]
]

performance.plot(kind="bar", figsize=(11, 6))
plt.title("Appliance Classification Performance")
plt.ylabel("Score")
plt.xlabel("Appliance")
plt.ylim(0, 1.1)
plt.xticks(rotation=30, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/classification_performance.png", dpi=300)
plt.show()

print("\nGraphs saved successfully!")
print("1. results/confusion_matrix.png")
print("2. results/feature_importance.png")
print("3. results/classification_performance.png")

print("\n========== VISUALIZATION COMPLETED ==========")