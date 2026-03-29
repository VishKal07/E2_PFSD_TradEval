import pandas as pd
<<<<<<< HEAD
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
=======
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
>>>>>>> 2a09392ae0b99a3752c7c750780483f44f55dcce
import joblib
import asyncio
import logging
from datetime import datetime, timedelta
import sys
import os

<<<<<<< HEAD
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
=======
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
>>>>>>> 2a09392ae0b99a3752c7c750780483f44f55dcce

from trading.services.data_ingestion import RealTimeDataIngestion
from ml.feature_engineering import FeatureEngineer
from ml.preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

<<<<<<< HEAD
class ModelTrainer:
    def __init__(self):
        self.ingestion = RealTimeDataIngestion()
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor()
    
    async def fetch_training_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch real training data from exchange"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching {days} days of real data for {symbol}")
        
        data = await self.ingestion.fetch_historical_data(symbol, start_date, end_date)
        
        if not data:
            raise Exception(f"No data fetched for {symbol}")
        
        df = pd.DataFrame(data)
        logger.info(f"Fetched {len(df)} data points")
        
        return df
    
    async def train(self, symbol: str = 'BTC/USDT', days: int = 30):
        """Train model with real-time data"""
        try:
            # Fetch real data
            df = await self.fetch_training_data(symbol, days)
            
            # Create features
            logger.info("Creating technical features...")
            df = self.feature_engineer.create_features(df)
            
            # Prepare labels (predict if price will increase next hour)
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Drop rows with NaN
            df = df.dropna()
            
            if len(df) < 100:
                raise Exception(f"Insufficient data: {len(df)} points")
            
            # Prepare features
            feature_cols = self.feature_engineer.get_feature_columns()
            X = df[feature_cols]
            y = df['target']
            
            # Scale features
            X_scaled = self.preprocessor.fit_transform(X, feature_cols)
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Train model
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            
            logger.info("Training model with time series cross-validation...")
            
            cv_scores = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                score = accuracy_score(y_val, model.predict(X_val))
                cv_scores.append(score)
                logger.info(f"Fold {fold+1} accuracy: {score:.4f}")
            
            # Train final model
            logger.info("Training final model...")
            model.fit(X_scaled, y)
            
            # Evaluate
            y_pred = model.predict(X_scaled)
            final_accuracy = accuracy_score(y, y_pred)
            
            logger.info(f"Training accuracy: {final_accuracy:.4f}")
            logger.info(f"CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
            
            # Save model
            os.makedirs('ml/model', exist_ok=True)
            joblib.dump(model, 'ml/model/risk_classifier.pkl')
            self.preprocessor.save('ml/model/scaler.pkl')
            
            # Save metadata
            metadata = {
                'symbol': symbol,
                'trained_date': datetime.now().isoformat(),
                'days_trained': days,
                'data_points': len(df),
                'train_accuracy': final_accuracy,
                'cv_accuracy_mean': np.mean(cv_scores),
                'cv_accuracy_std': np.std(cv_scores),
                'features': feature_cols
            }
            
            joblib.dump(metadata, 'ml/model/metadata.pkl')
            
            # Store in MongoDB
            try:
                from backend.models import TrainedModel
                TrainedModel.objects.create(
                    version=f"{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    symbol=symbol,
                    model_path='ml/model/risk_classifier.pkl',
                    train_accuracy=final_accuracy,
                    test_accuracy=np.mean(cv_scores),
                    is_active=True,
                    data_points=len(df)
                )
                logger.info("Model metadata stored in MongoDB")
            except Exception as e:
                logger.error(f"Failed to store in MongoDB: {e}")
            
            logger.info("✅ Model training complete!")
            return metadata
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            await self.ingestion.close()
=======
# ── Save ───────────────────────────────────────────────────────────
joblib.dump(model,  MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

size_kb = MODEL_PATH.stat().st_size / 1024
print(f"Model  saved → {MODEL_PATH} ({size_kb:.1f} KB)")
print(f"Scaler saved → {SCALER_PATH}")
>>>>>>> 2a09392ae0b99a3752c7c750780483f44f55dcce
