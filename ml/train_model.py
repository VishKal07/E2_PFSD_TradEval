"""
TradeEval – Risk Classifier Training Script
Run from the ml/ directory:  python train_model.py
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# ── Paths ─────────────────────────────────────────────────────────
ML_DIR     = Path(__file__).resolve().parent
DATA_DIR   = ML_DIR.parent / "Data"
MODEL_DIR  = ML_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH    = MODEL_DIR / "risk_classifier.pkl"
SCALER_PATH   = MODEL_DIR / "scaler.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# ── MUST match FEATURE_NAMES in risk_model.py exactly ─────────────
FEATURES = [
    "volatility",
    "max_drawdown",
    "sharpe_ratio",
    "volume_ratio",
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
print(f"Total rows loaded: {len(data)}")
print(f"Columns: {list(data.columns[:10])} ...")

if "close" not in data.columns:
    print(f"ERROR: No 'close' column. Found: {list(data.columns)}")
    sys.exit(1)

# ── Sort & clean ───────────────────────────────────────────────────
sort_cols = ["symbol", "date"] if "date" in data.columns else ["symbol"]
data = data.sort_values(sort_cols).reset_index(drop=True)
data["close"] = pd.to_numeric(data["close"], errors="coerce")

has_volume = "volume" in data.columns
if has_volume:
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce")

data.dropna(subset=["close"], inplace=True)

# ── Feature engineering ────────────────────────────────────────────
WINDOW = 20

def compute_features(group):
    g = group.copy()
    g["return_1d"] = g["close"].pct_change()

    # annualised volatility
    g["volatility"] = (
        g["return_1d"].rolling(WINDOW, min_periods=5).std() * np.sqrt(252)
    )

    # avg daily return
    g["avg_daily_return"] = (
        g["return_1d"].rolling(WINDOW, min_periods=5).mean()
    )

    # max drawdown over rolling window
    def rolling_drawdown(returns):
        cum  = (1 + returns).cumprod()
        peak = cum.cummax()
        return ((cum - peak) / peak).min()

    g["max_drawdown"] = g["return_1d"].rolling(WINDOW, min_periods=5).apply(
        rolling_drawdown, raw=False
    )

    # annualised Sharpe ratio
    rolling_mean = g["return_1d"].rolling(WINDOW, min_periods=5).mean()
    rolling_std  = g["return_1d"].rolling(WINDOW, min_periods=5).std()
    g["sharpe_ratio"] = (
        rolling_mean / rolling_std.replace(0, np.nan)
    ) * np.sqrt(252)

    # volume ratio
    if has_volume:
        avg_vol = g["volume"].rolling(WINDOW, min_periods=5).mean()
        g["volume_ratio"] = g["volume"] / avg_vol.replace(0, np.nan)
    else:
        g["volume_ratio"] = 1.0

    return g

print("Engineering features (this may take a moment)...")
data = data.groupby("symbol", group_keys=False).apply(compute_features)
data.dropna(subset=FEATURES, inplace=True)
print(f"Rows after feature engineering: {len(data)}")

if len(data) < 100:
    print("ERROR: Not enough data rows to train.")
    sys.exit(1)

# ── Risk labels — based on returns + drawdown, NOT volatility ──────
# This prevents the model from just memorising the label definition
return_33          = data["avg_daily_return"].quantile(0.33)
return_66          = data["avg_daily_return"].quantile(0.66)
drawdown_threshold = data["max_drawdown"].quantile(0.33)

def label_risk(row):
    bad_drawdown = row["max_drawdown"] < drawdown_threshold
    negative_ret = row["avg_daily_return"] < return_33
    positive_ret = row["avg_daily_return"] > return_66

    if bad_drawdown and negative_ret:
        return 2   # High
    elif positive_ret and not bad_drawdown:
        return 0   # Low
    else:
        return 1   # Medium

data["risk_label"] = data.apply(label_risk, axis=1)

X = data[FEATURES].values
y = data["risk_label"].values

print(f"\nTraining samples   : {len(X)}")
print(f"Class distribution : Low={int(np.sum(y==0))}  "
      f"Medium={int(np.sum(y==1))}  High={int(np.sum(y==2))}")

# ── Scale ──────────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print("\n── Classification Report ─────────────────────")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="f1_macro")
print(f"5-fold CV F1 (macro): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── Feature importances ────────────────────────────────────────────
print("\n── Feature Importances ───────────────────────")
for name, imp in sorted(
    zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]
):
    bar = "█" * int(imp * 40)
    print(f"  {name:<20} {imp:.4f}  {bar}")

# ── Save model + scaler + metadata ────────────────────────────────
joblib.dump(model,  MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

metadata = {
    "trained_at":        datetime.now().isoformat(),
    "features":          FEATURES,
    "num_features":      len(FEATURES),
    "training_samples":  int(len(X_train)),
    "test_samples":      int(len(X_test)),
    "cv_f1_mean":        round(float(cv_scores.mean()), 4),
    "cv_f1_std":         round(float(cv_scores.std()),  4),
    "class_distribution": {
        "Low":    int(np.sum(y == 0)),
        "Medium": int(np.sum(y == 1)),
        "High":   int(np.sum(y == 2)),
    },
    "model_params": model.get_params(),
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

size_kb = MODEL_PATH.stat().st_size / 1024
print(f"\nModel    saved  →  {MODEL_PATH}  ({size_kb:.1f} KB)")
print(f"Scaler   saved  →  {SCALER_PATH}")
print(f"Metadata saved  →  {METADATA_PATH}")
