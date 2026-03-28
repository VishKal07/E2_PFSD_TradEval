import pandas as pd


def sma_strategy(prices):

    df = pd.DataFrame(prices, columns=["price"])

    df["sma20"] = df["price"].rolling(20).mean()
    df["sma50"] = df["price"].rolling(50).mean()

    signal = "HOLD"

    if df["sma20"].iloc[-1] > df["sma50"].iloc[-1]:
        signal = "BUY"

    elif df["sma20"].iloc[-1] < df["sma50"].iloc[-1]:
        signal = "SELL"

    return signal