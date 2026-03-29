import pandas as pd
import numpy as np
from typing import List

class FeatureEngineer:
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create technical features for ML"""
        df = df.copy()
        
        # Price returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Price momentum
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)
        
        # Price position
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
        
        # Clean NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='bfill').fillna(method='ffill')
        df = df.fillna(0)
        
        return df
    
    @staticmethod
    def get_feature_columns() -> List[str]:
        """Return list of feature columns"""
        return [
            'returns', 'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
            'macd_histogram', 'bb_width', 'bb_position', 'volume_ratio',
            'volatility', 'atr', 'momentum_5', 'momentum_10',
            'momentum_20', 'price_position'
        ]