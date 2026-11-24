import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("serviceAccountKey.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

print("Firebase connected successfully!")

docs = db.collection("readings").limit(3).stream()

for doc in docs:
    print(doc.id, doc.to_dict())

print("Firebase verification completed!")