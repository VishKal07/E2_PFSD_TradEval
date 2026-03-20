"""
TradeEval – Risk Classifier Training Script
Run from the ml/ directory:  python train_model.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
ML_DIR    = Path(__file__).resolve().parent
DATA_DIR  = ML_DIR.parent / "Data"
MODEL_DIR = ML_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH  = MODEL_DIR / "risk_classifier.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

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

print(f"Loading {len(csv_files)} stock CSVs ...")

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
print(f"Total rows loaded: {len(data)}")

# ── Normalise column names ─────────────────────────────────────────
data.columns = [c.strip().lower() for c in data.columns]
print(f"Columns: {list(data.columns[:10])} ...")

if "close" not in data.columns:
    print(f"ERROR: No 'close' column. Available: {list(data.columns)}")
    sys.exit(1)

# ── Sort & clean ───────────────────────────────────────────────────
sort_cols = ["symbol", "date"] if "date" in data.columns else ["symbol"]
data = data.sort_values(sort_cols).reset_index(drop=True)
data["close"] = pd.to_numeric(data["close"], errors="coerce")
data.dropna(subset=["close"], inplace=True)

# ── Feature engineering ────────────────────────────────────────────
data["return_1d"]  = data.groupby("symbol")["close"].pct_change(1)
data["return_5d"]  = data.groupby("symbol")["close"].pct_change(5)
data["return_20d"] = data.groupby("symbol")["close"].pct_change(20)
data["volatility"] = data.groupby("symbol")["return_1d"].transform(
    lambda x: x.rolling(20, min_periods=5).std()
)

data.dropna(subset=["return_1d", "return_5d", "return_20d", "volatility"], inplace=True)
print(f"Rows after feature engineering: {len(data)}")

if len(data) < 100:
    print("ERROR: Not enough data rows to train.")
    sys.exit(1)

# ── Risk labels ────────────────────────────────────────────────────
vol_33 = data["volatility"].quantile(0.33)
vol_66 = data["volatility"].quantile(0.66)

def label_risk(v):
    if v <= vol_33:   return 0  # Low
    elif v <= vol_66: return 1  # Medium
    else:             return 2  # High

data["risk_label"] = data["volatility"].apply(label_risk)

FEATURES = ["return_1d", "return_5d", "return_20d", "volatility"]
X = data[FEATURES].values
y = data["risk_label"].values

print(f"Training samples  : {len(X)}")
print(f"Class distribution: Low={int(np.sum(y==0))}  Medium={int(np.sum(y==1))}  High={int(np.sum(y==2))}")

# ── Train ──────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n── Classification Report ─────────────────────")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

# ── Save ───────────────────────────────────────────────────────────
joblib.dump(model,  MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

size_kb = MODEL_PATH.stat().st_size / 1024
print(f"Model  saved → {MODEL_PATH} ({size_kb:.1f} KB)")
print(f"Scaler saved → {SCALER_PATH}")
