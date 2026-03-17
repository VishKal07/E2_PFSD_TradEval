import joblib
import numpy as np
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parents[3]
MODEL_PATH    = BASE_DIR / "ml" / "model" / "risk_classifier.pkl"
SCALER_PATH   = BASE_DIR / "ml" / "model" / "scaler.pkl"
META_PATH     = BASE_DIR / "ml" / "model" / "model_metadata.json"

# ── MUST match FEATURES in train_model.py exactly ─────────────────
FEATURE_NAMES = [
    "max_drawdown",
    "volume_ratio",
    "high_low_range",
    "close_position",
]
EXPECTED_FEATURES = 4
EXPECTED_FEATURES = len(FEATURE_NAMES)  # 4
RISK_LABELS       = {0: "Low", 1: "Medium", 2: "High"}

# ── lazy-loaded globals ────────────────────────────────────────────
_model  = None
_scaler = None


def _load_artifact(path: Path, name: str):
    if not path.exists():
        print(f"[risk_model] WARNING: {name} not found at {path}")
        return None
    try:
        obj = joblib.load(path)
        print(f"[risk_model] Loaded {name}")
        return obj
    except Exception as e:
        print(f"[risk_model] ERROR loading {name}: {e}")
        return None


def _get_model():
    global _model
    if _model is None:
        _model = _load_artifact(MODEL_PATH, "risk_classifier")
    return _model


def _get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = _load_artifact(SCALER_PATH, "scaler")
    return _scaler


def get_model_info() -> dict:
    """Return metadata about the trained model."""
    import json
    if META_PATH.exists():
        with open(META_PATH) as f:
            return json.load(f)
    return {"note": "No metadata found. Run: cd ml && python train_model.py"}


def classify_risk(features: list) -> dict:
    """
    Predict risk level from numeric features.

    Expected input — list of 4 floats in this exact order:
        [volatility, max_drawdown, sharpe_ratio, volume_ratio]

    Returns:
        risk_level   : 0=Low  1=Medium  2=High
        risk_label   : "Low" / "Medium" / "High"
        confidence   : float 0.0–1.0
        probabilities: per-class breakdown
        features_used: labelled input values
    """
    model = _get_model()

    if model is None:
        return {
            "error":      "Model not loaded. Run: cd ml && python train_model.py",
            "risk_level": -1,
            "confidence": 0.0,
        }

    # ── validate input ─────────────────────────────────────────────
    if not isinstance(features, list):
        return {
            "error":      f"features must be a list, got {type(features).__name__}",
            "risk_level": -1,
            "confidence": 0.0,
        }

    if len(features) != EXPECTED_FEATURES:
        return {
            "error": (
                f"Expected {EXPECTED_FEATURES} features {FEATURE_NAMES}, "
                f"got {len(features)}"
            ),
            "risk_level": -1,
            "confidence": 0.0,
        }

    try:
        arr = np.array(features, dtype=float).reshape(1, -1)
    except (ValueError, TypeError):
        return {
            "error":      "All features must be numeric",
            "risk_level": -1,
            "confidence": 0.0,
        }

    # ── scale if scaler available ──────────────────────────────────
    scaler = _get_scaler()
    if scaler is not None:
        arr = scaler.transform(arr)
    else:
        print("[risk_model] WARNING: No scaler — predictions may be inaccurate")

    # ── predict ────────────────────────────────────────────────────
    try:
        prediction    = int(model.predict(arr)[0])
        probabilities = model.predict_proba(arr)[0]
        confidence    = round(float(max(probabilities)), 4)

        classes   = model.clas
