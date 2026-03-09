import joblib
import numpy as np
from pathlib import Path

# Resolve project root: services -> api -> backend -> Project
BASE_DIR = Path(__file__).resolve().parents[3]

# Correct path: Project/ml/model/risk_classifier.pkl
MODEL_PATH = BASE_DIR / "ml" / "model" / "risk_classifier.pkl"

# Load model safely
model = None

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[risk_model] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[risk_model] Failed to load model: {e}")
        model = None
else:
    print(f"[risk_model] WARNING: Model not found at {MODEL_PATH}")
    print(f"[risk_model] Run: cd ml && python train_model.py")


def classify_risk(features):
    """
    Predict risk level from a list of numeric features.
    Returns a dict with risk_level and confidence.
    """
    if model is None:
        return {
            "risk_level": -1,
            "confidence": 0.0,
            "error": "Model not loaded. Run train_model.py first."
        }

    try:
        features_array = np.array(features).reshape(1, -1)
        prediction = int(model.predict(features_array)[0])
        probabilities = model.predict_proba(features_array)[0]
        confidence = float(max(probabilities))

        risk_labels = {0: "Low", 1: "Medium", 2: "High"}
        label = risk_labels.get(prediction, "Unknown")

        return {
            "risk_level": prediction,
            "risk_label": label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return {
            "risk_level": -1,
            "confidence": 0.0,
            "error": str(e)
        }
