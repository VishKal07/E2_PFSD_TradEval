import numpy as np


def optimize_portfolio(returns):

    weights = np.array(returns)

    weights = weights / np.sum(weights)

    return weights.tolist()