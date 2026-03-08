import os
import joblib

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../ml/model/risk_classifier.pkl"
)

model = joblib.load(MODEL_PATH)

def classify_risk(metrics):
    X = [[
        metrics["average_return"],
        metrics["volatility"],
        metrics["max_drawdown"],
        metrics["sharpe_ratio"],
    ]]
    return model.predict(X)[0]