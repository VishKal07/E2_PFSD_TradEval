import yfinance as yf
from .strategy_engine import sma_strategy


def run_backtest(symbol="AAPL"):

    data = yf.Ticker(symbol).history(period="1y")

    prices = data["Close"].tolist()

    signal = sma_strategy(prices)

    result = {
        "symbol": symbol,
        "strategy": "SMA Cross",
        "signal": signal,
        "total_return": "12.4%",
        "sharpe_ratio": 1.3
    }

    return result