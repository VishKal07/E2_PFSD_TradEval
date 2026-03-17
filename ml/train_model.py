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

WINDOW = 20

# ── Features — zero overlap with label definition ─────────────────
# Labels are defined by VOLATILITY REGIME only
# So volatility cannot be a feature — everything else is safe
FEATURES = [
    "max_drawdown",    # worst loss in window
    "volume_ratio",    # unusual trading activity
    "high_low_range",  # intraday price range as % of close
    "close_position",  # where close sits between high and low (0-1)
]

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

print(f"Found {len(csv_files)} stock CSVs")

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
print(f"Columns found: {list(data.columns[:10])}")

# check required columns
required = ["close"]
missing  = [c for c in required if c not in data.columns]
if missing:
    print(f"ERROR: Missing columns: {missing}")
    sys.exit(1)

has_volume   = "volume" in data.columns
has_high_low = "high" in data.columns and "low" in data.columns

sort_cols = ["symbol", "date"] if "date" in data.columns else ["symbol"]
data      = data.sort_values(sort_cols).reset_index(drop=True)

for col in ["close", "high", "low", "volume"]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

data.dropna(subset=["close"], inplace=True)
print(f"Total rows: {len(data)}")

# ── Feature engineering ────────────────────────────────────────────
def compute_features(g):
    g = g.copy()
    r = g["close"].pct_change()

    # volatility — used ONLY for labeling, not as a feature
    g["volatility"] = r.rolling(WINDOW, min_periods=5).std() * np.sqrt(252)

    # max drawdown — independent of return direction
    def dd(x):
        c = (1 + x).cumprod()
        return ((c - c.cummax()) / c.cummax()).min()
    g["max_drawdown"] = r.rolling(WINDOW, min_periods=5).apply(dd, raw=False)

    # volume ratio — is volume unusually high?
    if has_volume:
        avg_vol = g["volume"].rolling(WINDOW, min_periods=5).mean()
        g["volume_ratio"] = g["volume"] / avg_vol.replace(0, np.nan)
    else:
        g["volume_ratio"] = 1.0

    # high-low range as % of close — measures intraday volatility
    if has_high_low:
        g["high_low_range"] = (g["high"] - g["low"]) / g["close"].replace(0, np.nan)
        g["high_low_range"] = g["high_low_range"].rolling(WINDOW, min_periods=5).mean()
        # where did close land between high and low? (0=at low, 1=at high)
        hl_diff = (g["high"] - g["low"]).replace(0, np.nan)
        g["close_position"] = ((g["close"] - g["low"]) / hl_diff).rolling(WINDOW, min_periods=5).mean()
    else:
        g["high_low_range"] = 0.02   # neutral fallback
        g["close_position"] = 0.5

    return g

print("Engineering features...")
data = data.groupby("symbol", group_keys=False).apply(compute_features)
data.dropna(subset=FEATURES + ["volatility"], inplace=True)
print(f"Rows after feature engineering: {len(data)}")

# ── Labels — volatility regime ONLY ───────────────────────────────
# We label by volatility but DON'T use volatility as a feature
# The model must infer risk from drawdown, volume, price structure
vol_33 = data["volatility"].quantile(0.33)
vol_66 = data["volatility"].quantile(0.66)

def label_risk(v):
    if v <= vol_33:   return 0  # Low
    elif v <= vol_66: return 1  # Medium
    else:             return 2  # High

data["risk_label"] = data["volatility"].apply(label_risk)

print(f"\nLabel distribution:")
print(f"  Low={int((data['risk_label']==0).sum())}  "
      f"Medium={int((data['risk_label']==1).sum())}  "
      f"High={int((data['risk_label']==2).sum())}")

# ── Split by STOCK SYMBOL not by time ─────────────────────────────
# This prevents the model from memorizing per-stock patterns
all_symbols  = data["symbol"].unique()
np.random.seed(42)
np.random.shuffle(all_symbols)

split_idx    = int(len(all_symbols) * 0.8)
train_syms   = set(all_symbols[:split_idx])
test_syms    = set(all_symbols[split_idx:])

train_data   = data[data["symbol"].isin(train_syms)]
test_data    = data[data["symbol"].isin(test_syms)]

X_train = train_data[FEATURES].values
y_train = train_data["risk_label"].values
X_test  = test_data[FEATURES].values
y_test  = test_data["risk_label"].values

print(f"\nTrain: {len(train_syms)} stocks, {len(X_train)} rows")
print(f"Test : {len(test_syms)} stocks, {len(X_test)} rows")
print(f"Test symbols (never seen in training): {sorted(test_syms)}")

# ── Scale ──────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Train ──────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=50,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── Evaluate on completely unseen stocks ──────────────────────────
y_pred = model.predict(X_test)
print("\n── Classification Report (unseen stocks) ────")
print(classification_report(y_test, y_pred, target_names=["Low","Medium","High"]))
f1 = f1_score(y_test, y_pred, average="macro")
print(f"Holdout F1 (macro): {f1:.3f}")

# ── Feature importances ────────────────────────────────────────────
print("\n── Feature Importances ──────────────────────")
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
    "train_stocks":      list(train_syms),
    "test_stocks":       list(test_syms),
    "training_rows":     int(len(X_train)),
    "test_rows":         int(len(X_test)),
    "holdout_f1":        round(float(f1), 4),
    "class_distribution": {
        "Low":    int(np.sum(y_train==0) + np.sum(y_test==0)),
        "Medium": int(np.sum(y_train==1) + np.sum(y_test==1)),
        "High":   int(np.sum(y_train==2) + np.sum(y_test==2)),
    },
}
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

size_kb = MODEL_PATH.stat().st_size / 1024
print(f"\nModel    saved → {MODEL_PATH} ({size_kb:.1f} KB)")
print(f"Scaler   saved → {SCALER_PATH}")
print(f"Metadata saved → {METADATA_PATH}")
