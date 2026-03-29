import numpy as np
import yfinance as yf
from .news_fetcher import fetch_news

def analyze_event(symbol: str) -> dict:

    if not symbol or not isinstance(symbol, str):
        return {"error": "Invalid symbol provided"}

    symbol = symbol.upper().strip()

    # ── 1. PRICE HISTORY ─────────────────────────────────────────
    try:
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period="3mo")

        if hist.empty:
            return {"error": f"No price data found for {symbol}"}

        prices           = hist["Close"].tolist()
        returns          = np.diff(prices) / prices[:-1]
        volatility       = round(float(np.std(returns) * np.sqrt(252) * 100), 2)
        recent_return    = round(((prices[-1] - prices[0]) / prices[0]) * 100, 2)
        avg_daily_return = round(float(np.mean(returns) * 100), 4)

    except Exception as e:
        return {"error": f"Price data fetch failed: {str(e)}"}

    # ── 2. NEWS + SENTIMENT ───────────────────────────────────────
    news_result = fetch_news(symbol, days_back=7)

    if "error" in news_result and not news_result.get("articles"):
        news_signal   = "neutral"
        avg_sentiment = 0.0
        news_error    = news_result["error"]
    else:
        news_signal   = news_result.get("overall_signal", "neutral")
        avg_sentiment = news_result.get("avg_score", 0.0)
        news_error    = None

    # ── 3. EARNINGS CALENDAR ─────────────────────────────────────
    earnings_info = {}
    try:
        cal = ticker.calendar
        if cal is not None and not cal.empty:
            earnings_info = {
                col: str(cal[col].iloc[0])
                for col in cal.columns
                if not cal[col].isnull().all()
            }
    except Exception:
        earnings_info = {"note": "No earnings calendar available"}

    # ── 4. COMBINED BEHAVIOR SIGNAL ──────────────────────────────
    if recent_return > 2 and news_signal == "bullish":
        behavior = "strongly bullish"
    elif recent_return < -2 and news_signal == "bearish":
        behavior = "strongly bearish"
    elif recent_return > 0 and news_signal in ("bullish", "neutral"):
        behavior = "mildly bullish"
    elif recent_return < 0 and news_signal in ("bearish", "neutral"):
        behavior = "mildly bearish"
    else:
        behavior = "mixed / uncertain"

    return {
        "symbol":              symbol,
        "volatility_pct":      volatility
