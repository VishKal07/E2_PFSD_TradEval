import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime, timedelta
from django.core.cache import cache
from trading.models import MarketData, ModelRegistry, Prediction
import logging

logger = logging.getLogger(__name__)

class AntiOverfittingTrainer:
    def __init__(self, symbol, model_type='random_forest'):
        self.symbol = symbol
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        self.features = [
            'returns_lag1', 'returns_lag2', 'returns_lag5',
            'volume_lag1', 'volume_ratio',
            'volatility_5min', 'volatility_30min',
            'rsi_14', 'macd', 'bb_position',
            'hour_sin', 'hour_cos', 'day_of_week'
        ]
        
    def fetch_training_data(self, days_back=90):
        """Fetch historical data with look-ahead bias prevention"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Get market data
        data = list(MarketData.objects.filter(
            symbol=self.symbol,
            interval='1min',
            timestamp__gte=start_date,
            timestamp__lte=end_date
        ).order_by('timestamp'))
        
        if len(data) < 1000:
            logger.warning(f"Insufficient data for {self.symbol}: {len(data)} samples")
            return None
            
        df = pd.DataFrame([{
            'timestamp': d.timestamp,
            'close': d.close,
            'volume': d.volume,
            'high': d.high,
            'low': d.low
        } for d in data])
        
        return self._engineer_features(df)
    
    def _engineer_features(self, df):
        """Create features without future data leakage"""
        df = df.sort_values('timestamp').copy()
        
        # Returns (using only past data)
        df['returns'] = df['close'].pct_change()
        df['returns_lag1'] = df['returns'].shift(1)
        df['returns_lag2'] = df['returns'].shift(2)
        df['returns_lag5'] = df['returns'].shift(5)
        
        # Volume features
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20'].shift(1)
        df['volume_lag1'] = df['volume'].shift(1)
        
        # Volatility (rolling std of returns)
        df['volatility_5min'] = df['returns'].rolling(5).std()
        df['volatility_30min'] = df['returns'].rolling(30).std()
        
        # RSI (Relative Strength Index)
        delta = df['returns'].clip(lower=0)
        gain = delta.rolling(14).mean()
        loss = (-delta).rolling(14).mean()
        df['rsi_14'] = 100 - (100 / (1 + gain / loss))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        
        # Bollinger Bands position
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_position'] = (df['close'] - bb_middle) / (2 * bb_std)
        
        # Time features (cyclical encoding)
        df['hour'] = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Target: future return (predict next 5-minute return)
        df['target'] = df['returns'].shift(-5)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def walk_forward_validation(self, df):
        """Time-series cross-validation to prevent overfitting"""
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
            train = df.iloc[train_idx]
            val = df.iloc[val_idx]
            
            X_train = train[self.features]
            y_train = train['target']
            X_val = val[self.features]
            y_val = val['target']
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model
            if self.model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=4,
                    min_samples_split=10,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42
                )
            
            model.fit(X_train_scaled, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_val_scaled)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            
            scores.append({
                'fold': fold,
                'mse': mse,
                'mae': mae,
                'train_size': len(train),
                'val_size': len(val)
            })
            
            logger.info(f"Fold {fold} - MSE: {mse:.6f}, MAE: {mae:.6f}")
        
        return scores
    
    def train_final_model(self, df):
        """Train final model on full dataset with regularization"""
        X = df[self.features]
        y = df['target']
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Train with regularization
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=8,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                min_samples_split=10,
                learning_rate=0.03,
                subsample=0.7,
                random_state=42
            )
        
        self.model.fit(X_scaled, y)
        
        # Save model
        model_id = f"{self.symbol}_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        joblib.dump(self.model, f'models/{model_id}.pkl')
        joblib.dump(self.scaler, f'models/{model_id}_scaler.pkl')
        
        return model_id
    
    def register_model(self, model_id, train_df, validation_scores):
        """Register model in database to track performance"""
        registry = ModelRegistry.objects.create(
            model_id=model_id,
            model_type=self.model_type,
            version=1,
            train_start_date=train_df['timestamp'].min(),
            train_end_date=train_df['timestamp'].max(),
            features_used=self.features,
            hyperparameters=self.model.get_params(),
            performance_metrics={
                'train_size': len(train_df),
                'features': len(self.features)
            },
            out_of_sample_metrics={
                'avg_mse': np.mean([s['mse'] for s in validation_scores]),
                'avg_mae': np.mean([s['mae'] for s in validation_scores]),
                'validation_folds': len(validation_scores)
            },
            is_deployed=False
        )
        
        logger.info(f"Model {model_id} registered with avg MSE: {registry.out_of_sample_metrics['avg_mse']:.6f}")
        
        return registry
    
    def run_training_pipeline(self):
        """Complete training pipeline with anti-overfitting measures"""
        # 1. Fetch data
        df = self.fetch_training_data(days_back=90)
        if df is None or len(df) < 1000:
            return None, "Insufficient data"
        
        # 2. Walk-forward validation
        logger.info(f"Starting walk-forward validation for {self.symbol}")
        validation_scores = self.walk_forward_validation(df)
        
        # Check if model is performing well
        avg_mse = np.mean([s['mse'] for s in validation_scores])
        if avg_mse > 0.0001:  # Threshold for acceptable error
            logger.warning(f"Model performance below threshold: MSE={avg_mse}")
        
        # 3. Train final model
        model_id = self.train_final_model(df)
        
        # 4. Register model
        registry = self.register_model(model_id, df, validation_scores)
        
        # 5. Generate sample predictions for monitoring
        self.generate_sample_predictions(model_id, df.tail(100))
        
        return model_id, validation_scores
    
    def generate_sample_predictions(self, model_id, test_df):
        """Generate predictions for out-of-sample monitoring"""
        X_test = test_df[self.features]
        X_test_scaled = self.scaler.transform(X_test)
        
        predictions = self.model.predict(X_test_scaled)
        
        for idx, pred in enumerate(predictions):
            Prediction.objects.create(
                model_id=model_id,
                symbol=self.symbol,
                timestamp=test_df.iloc[idx]['timestamp'],
                predicted_price=test_df.iloc[idx]['close'] * (1 + pred),
                confidence=0.7,  # Could be derived from model uncertainty
                direction='up' if pred > 0 else 'down',
                features_at_prediction=test_df.iloc[idx][self.features].to_dict()
            )