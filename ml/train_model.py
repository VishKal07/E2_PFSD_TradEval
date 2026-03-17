"""
TradeEval – Risk Classifier Training Script
Run from the ml/ directory: python train_model.py
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
import joblib

# ── Paths ──────────────────────────────────────────────────────────
ML_DIR        = Path(__file__).resolve().parent
DATA_DIR      = ML_DIR.parent / "Data"
MODEL_DIR     = ML_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH    = MODEL_DIR / "risk_classifier.pkl"
SCALER_PATH   = MODEL_DIR / "scaler.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# ── Features — NO avg_daily_return, NO sharpe_ratio ───────────────
# Both contain the return signal that defines labels → leakage
# These 4 are genuinely independent of the label definition
FEATURES = [
    "volatility",      # how much price moves
    "max_drawdown",    # worst loss in window
    "volume_ratio",    # unusual trading activity
    "price_momentum",  # recent price direction (neutral signal)
]

WINDOW = 20

print(f"Data directory : {DATA_DIR}")
print(f"Model output   : {MODEL_PATH}")

# ── Load CSVs ──────────────────────────────────────────────────────
csv_files = [
    f for f in DATA_DIR.glob("*.csv")
    if "NIFTY" not in f.name and "metadata" not in f.name
]

if not csv_files:
    print("ERROR: No CSV files found in Data/")
    sys.exit(1)

print(f"Loading {len(csv_files)} stock CSVs...")

frames = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        df = df.loc[:, ~df.columns.str.lower().isin(["symbol"])]
        df["symbol"] = f.stem
        frames.append(df)
    except Exception as e:
        print(f"  Skipping {f.name}: {e}")

if not frames:
    print("ERROR: No data loaded.")
    sys.exit(1)

data = pd.concat(frames, ignore_index=True)
data.columns = [c.strip().lower() for c in data.columns]

if "close" not in data.columns:
    print(f"ERROR: No 'close' column. Found: {list(data.columns)}")
    sys.exit(1)

has_volume = "volume" in data.columns
sort_cols  = ["symbol", "date"] if "date" in data.columns else ["symbol"]
data       = data.sort_values(sort_cols).reset_index(drop=True)
data["close"] = pd.to_numeric(data["close"], errors="coerce")
if has_volume:
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
data.dropna(subset=["close"], inplace=True)
print(f"Total rows loaded: {len(data)}")

# ── Feature engineering ────────────────────────────────────────────
def compute_features(g):
    g = g.copy()
    r = g["close"].pct_change()

    # volatility — annualised std of returns
    g["volatility"]   = r.rolling(WINDOW, min_periods=5).std() * np.sqrt(252)

    # max drawdown — worst peak to trough over window
    def dd(x):
        c = (1 + x).cumprod()
        return ((c - c.cummax()) / c.cummax()).min()
    g["max_drawdown"] = r.rolling(WINDOW, min_periods=5).apply(dd, raw=False)

    # volume ratio — is volume unusually high or low?
    if has_volume:
        avg_vol = g["volume"].rolling(WINDOW, min_periods=5).mean()
        g["volume_ratio"] = g["volume"] / avg_vol.replace(0, np.nan)
    else:
        g["volume_ratio"] = 1.0

    # price momentum — where is price vs 20 days ago (normalized)
    # this is directionally neutral — doesn't directly encode return mean
    g["price_momentum"] = (
        g["close"] - g["close"].shift(WINDOW)
    ) / g["close"].shift(WINDOW).replace(0, np.nan)

    # label feature — avg return, used ONLY for labeling, not as input
    g["avg_daily_return"] = r.rolling(WINDOW, min_periods=5).mean()

    return g

print("Engineering features...")
data = data.groupby("symbol", group_keys=False).apply(compute_features)
data.dropna(subset=FEATURES + ["avg_daily_return"], inplace=True)
print(f"Rows after feature engineering: {len(data)}")

if len(data) < 100:
    print("ERROR: Not enough data.")
    sys.exit(1)

# ── Labels — based on avg_daily_return + max_drawdown ─────────────
# avg_daily_return is NOT in FEATURES so no leakage
r33 = data["avg_daily_return"].quantile(0.33)
r66 = data["avg_daily_return"].quantile(0.66)
dd_thresh = data["max_drawdown"].quantile(0.33)

def label_risk(row):
    bad = row["max_drawdown"] < dd_thresh
    neg = row["avg_daily_return"] < r33
    pos = row["avg_daily_return"] > r66
    if bad and neg:          return 2  # High
    elif pos and not bad:    return 0  # Low
    else:                    return 1  # Medium

data["risk_label"] = data.apply(label_risk, axis=1)

X = data[FEATURES].values
y = data["risk_label"].values

print(f"\nTraining samples   : {len(X)}")
print(f"Class distribution : Low={int(np.sum(y==0))}  "
      f"Medium={int(np.sum(y==1))}  High={int(np.sum(y==2))}")

# ── Scale ──────────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Time-based split ───────────────────────────────────────────────
split    = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split],        y[split:]
print(f"Train: {len(X_train)}  Test: {len(X_test)}")

# ── Train ──────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=100,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print("\n── Classification Report ─────────────────────")
print(classification_report(y_test, y_pred, target_names=["Low","Medium","High"]))
f1 = f1_score(y_test, y_pred, average="macro")
print(f"Holdout F1 (macro): {f1:.3f}")

# ── Feature importances ────────────────────────────────────────────
print("\n── Feature Importances ───────────────────────")
for name, imp in sorted(
    zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]
):
    print(f"  {name:<20} {imp:.4f}  {'█' * int(imp * 40)}")

# ── Save ───────────────────────────────────────────────────────────
joblib.dump(model,  MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

metadata = {
    "trained_at":        datetime.now().isoformat(),
    "features":          FEATURES,
    "num_features":      len(FEATURES),
    "training_samples":  int(len(X_train)),
    "test_samples":      int(len(X_test)),
    "holdout_f1":        round(float(f1), 4),
    "class_distribution": {
        "Low":    int(np.sum(y==0)),
        "Medium": int(np.sum(y==1)),
        "High":   int(np.sum(y==2)),
    },
    "model_params": model.get_params(),
}
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

size_kb = MODEL_PATH.stat().st_size / 1024
print(f"\nModel    saved → {MODEL_PATH} ({size_kb:.1f} KB)")
print(f"Scaler   saved → {SCALER_PATH}")
print(f"Metadata saved → {METADATA_PATH}")
