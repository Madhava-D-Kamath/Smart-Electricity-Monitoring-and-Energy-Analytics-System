import firebase_admin
from firebase_admin import credentials, firestore

import joblib
import pandas as pd
from datetime import datetime

# ==========================================
# 1. FIREBASE CONNECTION
# ==========================================

if not firebase_admin._apps:

    cred = credentials.Certificate("serviceAccountKey.json")

    firebase_admin.initialize_app(cred)

db = firestore.client()

print("\n[FIREBASE] Connected successfully!")

# ==========================================
# 2. LOAD RANDOM FOREST MODEL
# ==========================================

model = joblib.load(
    "models/random_forest_appliance.pkl"
)

print("[ML] Random Forest model loaded!")

# ==========================================
# 3. FETCH LATEST FIREBASE READING
# ==========================================

docs = (
    db.collection("readings")
    .order_by(
        "timestamp",
        direction=firestore.Query.DESCENDING
    )
    .limit(1)
    .stream()
)

latest_reading = None

for doc in docs:
    latest_reading = doc.to_dict()

if latest_reading is None:
    print("[ERROR] No Firebase readings found!")
    raise SystemExit

print("\n[FIREBASE] Latest reading received!")

# ==========================================
# 4. EXTRACT SENSOR VALUES
# ==========================================

voltage = float(latest_reading["voltage_V"])

current = float(latest_reading["current_A"])

power = float(latest_reading["power_W"])

frequency = float(latest_reading["frequency_Hz"])

power_factor = float(latest_reading["power_factor"])

# ==========================================
# 5. PREPARE MODEL INPUT
# ==========================================

features = pd.DataFrame([{

    "voltage_V": voltage,

    "current_A": current,

    "power_W": power,

    "frequency_Hz": frequency,

    "power_factor": power_factor

}])

print("\n[ML] Input features:")
print(features)

# ==========================================
# 6. APPLIANCE PREDICTION
# ==========================================

prediction = model.predict(features)[0]

probabilities = model.predict_proba(features)[0]

confidence = max(probabilities) * 100

# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n====================================")

print(" AI SMART ELECTRICITY MONITORING")

print("====================================")

print(f"Voltage: {voltage} V")

print(f"Current: {current} A")

print(f"Power: {power} W")

print(f"Frequency: {frequency} Hz")

print(f"Power Factor: {power_factor}")

print("------------------------------------")

print("PREDICTED APPLIANCE:", prediction)

print("CONFIDENCE:", round(confidence, 2), "%")

print("====================================")
