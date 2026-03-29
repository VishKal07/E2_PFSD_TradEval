import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_columns = None
    
    def fit(self, df: pd.DataFrame, feature_cols: list):
        """Fit scaler on training data"""
        self.feature_columns = feature_cols
        self.scaler.fit(df[feature_cols])
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler"""
        if self.feature_columns is None:
            raise ValueError("Preprocessor not fitted yet")
        
        scaled_data = self.scaler.transform(df[self.feature_columns])
        return pd.DataFrame(scaled_data, columns=self.feature_columns, index=df.index)
    
    def fit_transform(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Fit and transform data"""
        self.fit(df, feature_cols)
        return self.transform(df)
    
    def save(self, path: str = 'ml/model/scaler.pkl'):
        """Save preprocessor"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, path)
    
    def load(self, path: str = 'ml/model/scaler.pkl'):
        """Load preprocessor"""
        if os.path.exists(path):
            data = joblib.load(path)
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']
            return True
        return False