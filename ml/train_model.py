import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib
import asyncio
import logging
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.services.data_ingestion import RealTimeDataIngestion
from ml.feature_engineering import FeatureEngineer
from ml.preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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