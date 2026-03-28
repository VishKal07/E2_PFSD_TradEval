import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "model", "risk_classifier.pkl")

model = joblib.load(MODEL_PATH)

def classify_risk(features):
    prediction = model.predict([features])[0]

    mapping = {
        0: "Low Risk",
        1: "Medium Risk",
        2: "High Risk"
    }

    return mapping.get(prediction, "Unknown")