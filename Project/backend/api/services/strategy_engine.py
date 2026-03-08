from .metrics import calculate_metrics

def run_strategy(symbol, strategy):
    # Dummy returns for now (replace later)
    returns = [0.02, -0.01, 0.03, -0.02, 0.01]

    metrics = calculate_metrics(returns)

    return {
        "symbol": symbol,
        "strategy": strategy,
        "metrics": metrics
    }