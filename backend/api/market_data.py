import yfinance as yf

def get_stock_data(symbol):

    ticker = yf.Ticker(symbol)

    data = ticker.history(period="6mo")

    prices = data["Close"].tolist()

    dates = data.index.strftime("%Y-%m-%d").tolist()

    return {
        "symbol": symbol,
        "prices": prices,
        "dates": dates
    }