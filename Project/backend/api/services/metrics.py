import numpy as np

def calculate_metrics(returns):
    r = np.array(returns)

    avg_return = r.mean()
    volatility = r.std()
    sharpe = avg_return / volatility if volatility != 0 else 0
    max_drawdown = r.min()

    return {
        "average_return": round(float(avg_return), 4),
        "volatility": round(float(volatility), 4),
        "sharpe_ratio": round(float(sharpe), 4),
        "max_drawdown": round(float(max_drawdown), 4),
    }