import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import yfinance as yf

print("Training Risk Prediction Model...")

# Generate synthetic training data based on real market patterns
np.random.seed(42)
n_samples = 5000

# Features: [return_1d, return_5d, return_20d, volatility, volume_ratio, rsi]
features = []
labels = []

# Get real data patterns from Yahoo Finance
try:
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="1y")
    returns = hist['Close'].pct_change().dropna()
    real_vol = returns.std()
    print(f"Real market volatility: {real_vol:.4f}")
except:
    real_vol = 0.02

for i in range(n_samples):
    # Generate realistic features
    ret_1d = np.random.normal(0, 0.02)
    ret_5d = np.random.normal(0, 0.04)
    ret_20d = np.random.normal(0, 0.06)
    volatility = abs(np.random.normal(real_vol, 0.01))
    volume_ratio = np.random.normal(1, 0.3)
    rsi = np.random.uniform(20, 80)
    
    features.append([ret_1d, ret_5d, ret_20d, volatility, volume_ratio, rsi])
    
    # Determine risk label based on features
    risk_score = (abs(ret_1d) * 10 + abs(ret_5d) * 5 + volatility * 50 + 
                  (1/volume_ratio if volume_ratio > 0 else 0) + 
                  (abs(rsi - 50) / 50))
    
    if risk_score > 1.5:
        label = 2  # High Risk
    elif risk_score > 0.7:
        label = 1  # Medium Risk
    else:
        label = 0  # Low Risk
    
    labels.append(label)

# Convert to arrays
X = np.array(features)
y = np.array(labels)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_scaled, y)

# Save model and scaler
os.makedirs('ml/model', exist_ok=True)
joblib.dump(model, 'ml/model/risk_classifier.pkl')
joblib.dump(scaler, 'ml/model/scaler.pkl')

print(f"✅ Model trained successfully!")
print(f"   Accuracy: {model.score(X_scaled, y):.2%}")
print(f"   Features: return_1d, return_5d, return_20d, volatility, volume_ratio, rsi")
print(f"   Model saved to: ml/model/risk_classifier.pkl")