import os
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURATION
# ==========================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

KEY_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", os.path.join(BASE_DIR, "serviceAccountKey.json"))

EXPECTED_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "SET_YOUR_FIREBASE_PROJECT_ID")
COLLECTION_NAME = "readings"

# ==========================================
# 2. FIREBASE CONNECTION
# ==========================================

try:

    if EXPECTED_PROJECT_ID == "SET_YOUR_FIREBASE_PROJECT_ID":
        raise ValueError("Set FIREBASE_PROJECT_ID in your .env file before starting the server.")

    cred = credentials.Certificate(KEY_PATH)

    if cred.project_id != EXPECTED_PROJECT_ID:
        raise ValueError(
            f"Firebase project mismatch: credentials belong to {cred.project_id}, "
            f"configured project is {EXPECTED_PROJECT_ID}"
        )

    if not firebase_admin._apps:

        firebase_admin.initialize_app(cred)

    db = firestore.client()

    print("\n================================")
    print("FIREBASE CONNECTION SUCCESSFUL")
    print("================================")
    print("Firebase Project ID:", cred.project_id)
    print("Firestore Collection:", COLLECTION_NAME)
    print("================================\n", flush=True)

except Exception as e:

    print("\n[FIREBASE ERROR]", str(e), flush=True)
    raise


# ==========================================
# 3. HOME ROUTE
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "firebase": "Connected",
        "project_id": cred.project_id,
        "collection": COLLECTION_NAME,
        "status": "Running"
    })


# ==========================================
# 4. RECEIVE ESP32 SENSOR DATA
# ==========================================

@app.route("/api/readings", methods=["POST"])
def receive_readings():

    try:

        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "status": "error",
                "message": "Invalid JSON data"
            }), 400

        print("\n================================")
        print("NEW SENSOR READING RECEIVED")
        print("================================")

        # Extract sensor values

        reading = {

            "node_id": data.get("node_id", "ESP32_01"),

            "appliance": data.get(
                "appliance", "unknown"
            ),

            "voltage_V": float(
                data.get("voltage_V", 0)
            ),

            "current_A": float(
                data.get("current_A", 0)
            ),

            "power_W": float(
                data.get("power_W", 0)
            ),

            "energy_kWh": float(
                data.get("energy_kWh", 0)
            ),

            "frequency_Hz": float(
                data.get("frequency_Hz", 0)
            ),

            "power_factor": float(
                data.get("power_factor", 0)
            ),

            "timestamp": datetime.now(timezone.utc)
        }

        # Print received values

        for key, value in reading.items():

            print(f"{key}: {value}")

        print("--------------------------------")

        # ==================================
        # 5. STORE DATA IN FIRESTORE
        # ==================================

        update_time, doc_ref = (
            db.collection(COLLECTION_NAME).add(reading)
        )

        print("\n[FIREBASE SUCCESS]")
        print("Firebase Project:", cred.project_id)
        print("Collection:", COLLECTION_NAME)
        print("Document ID:", doc_ref.id)
        print("Write Time:", update_time)
        print("Data successfully stored!")
        print("================================\n", flush=True)

        return jsonify({

            "status": "success",

            "message": "Data stored in Firebase",

            "document_id": doc_ref.id,

            "project_id": EXPECTED_PROJECT_ID

        }), 200

    except Exception as e:

        print("\n[FIREBASE WRITE ERROR]")
        print(str(e))
        print("================================\n", flush=True)

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# ==========================================
# 6. START FLASK SERVER
# ==========================================

if __name__ == "__main__":

    print("\n================================")
    print("AI SMART ELECTRICITY MONITORING")
    print("Flask Server Starting...")
    print("================================\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )