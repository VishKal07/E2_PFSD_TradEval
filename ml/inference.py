import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_engineering import FeatureEngineer
from ml.preprocessing import DataPreprocessor
from trading.services.data_ingestion import RealTimeDataIngestion

logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self):
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.metadata = None
        self._load_model()
    
    def _load_model(self):
        """Load the latest trained model"""
        try:
            if os.path.exists('ml/model/risk_classifier.pkl'):
                self.model = joblib.load('ml/model/risk_classifier.pkl')
                self.preprocessor.load('ml/model/scaler.pkl')
                
                if os.path.exists('ml/model/metadata.pkl'):
                    self.metadata = joblib.load('ml/model/metadata.pkl')
                
                logger.info("Model loaded successfully")
                return True
            else:
                logger.warning("No model found")
                return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            return False
    
    def predict(self, symbol: str = 'BTC/USDT') -> dict:
        """Make prediction for a symbol"""
        if self.model is None:
            if not self._load_model():
                return {'error': 'No model available', 'success': False}
        
        try:
            # Fetch recent data
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            ingestion = RealTimeDataIngestion()
            data = loop.run_until_complete(ingestion.fetch_realtime_data(symbol, limit=100))
            loop.close()
            
            if not data:
                return {'error': 'No data available', 'success': False}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df = self.feature_engineer.create_features(df)
            
            # Get latest features
            feature_cols = self.feature_engineer.get_feature_columns()
            latest_features = df[feature_cols].iloc[-1:].copy()
            
            # Scale features
            scaled_features = self.preprocessor.transform(latest_features)
            
            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]
            confidence = float(max(probabilities))
            
            current_price = float(df['close'].iloc[-1])
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'prediction': int(prediction),
                'prediction_label': 'UP' if prediction == 1 else 'DOWN',
                'confidence': confidence,
                'probabilities': {
                    'down': float(probabilities[0]),
                    'up': float(probabilities[1])
                },
                'model_version': self.metadata.get('symbol', 'unknown') if self.metadata else 'unknown',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {'error': str(e), 'success': False}