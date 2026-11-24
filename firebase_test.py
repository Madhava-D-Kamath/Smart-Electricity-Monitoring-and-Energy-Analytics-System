import firebase_admin
from firebase_admin import credentials, firestore
from pathlib import Path

# Locate Firebase credentials
key_path = Path(__file__).parent / "serviceAccountKey.json"

# Connect to Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(str(key_path))
    firebase_admin.initialize_app(cred)

db = firestore.client()

print("Firebase connected successfully!")

# Read the latest document
docs = (
    db.collection("readings")
    .limit(5)
    .stream()
)

print("\nFirebase readings:")

for doc in docs:
    print(doc.to_dict())