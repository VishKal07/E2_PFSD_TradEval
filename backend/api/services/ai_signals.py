from .strategy_engine import sma_strategy


def generate_ai_signal(prices):

    signal = sma_strategy(prices)

    return {
        "signal": signal,
        "confidence": 0.72
    }