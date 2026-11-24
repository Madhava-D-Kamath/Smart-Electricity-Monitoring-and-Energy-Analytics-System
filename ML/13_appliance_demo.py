import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("\n" + "=" * 55)
print("       AI SMART ELECTRICITY MONITORING SYSTEM")
print("=" * 55)

# STEP 1: Load the existing dataset
print("\nLoading dataset...")

df = pd.read_csv("results/ml_ready_dataset.csv")

# Exclude uncertain Session 7
df = df[df["session_id"] != 7].copy()

print("Dataset loaded successfully!")
print("Total readings:", len(df))

# STEP 2: Select electrical parameters
features = [
    "voltage_V",
    "current_A",
    "power_W",
    "frequency_Hz",
    "power_factor"
]

X = df[features]
y = df["appliance"]

# STEP 3: Recreate the original test dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# STEP 4: Load trained Random Forest model
print("\nLoading trained AI model...")

model = joblib.load("models/random_forest_appliance.pkl")

print("AI model loaded successfully!")

# STEP 5: Calculate test accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Test Accuracy:", round(accuracy * 100, 2), "%")

# STEP 6: Select 10 sample readings
samples = X_test.sample(10, random_state=42)

print("\n" + "=" * 55)
print("             LIVE-STYLE AI DEMONSTRATION")
print("=" * 55)

correct = 0

for i, (index, reading) in enumerate(samples.iterrows(), 1):

    input_data = pd.DataFrame(
        [reading],
        columns=features
    )

    # Predict appliance
    prediction = model.predict(input_data)[0]

    # Calculate prediction confidence
    probabilities = model.predict_proba(input_data)[0]

    confidence = max(probabilities) * 100

    actual = y.loc[index]

    print("\nReading:", i)
    print("-" * 45)

    print("Voltage          :", reading["voltage_V"], "V")
    print("Current          :", reading["current_A"], "A")
    print("Power            :", reading["power_W"], "W")
    print("Power Factor     :", reading["power_factor"])
    print("Frequency        :", reading["frequency_Hz"], "Hz")

    print("\nActual Appliance :", actual)
    print("AI Prediction    :", prediction)
    print("Confidence       :", round(confidence, 2), "%")

    if prediction == actual:
        print("Result           : CORRECT")
        correct += 1
    else:
        print("Result           : INCORRECT")

# STEP 7: Final summary
print("\n" + "=" * 55)
print("               DEMONSTRATION SUMMARY")
print("=" * 55)

print("Total readings tested :", len(samples))
print("Correct predictions   :", correct)
print("Incorrect predictions :", len(samples) - correct)

print(
    "Sample prediction accuracy:",
    round((correct / len(samples)) * 100, 2),
    "%"
)

print("\nAI Appliance Identification Completed!")
print("=" * 55)