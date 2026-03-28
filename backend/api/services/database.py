import os
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "tradeeval")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

users = db["users"]
trades = db["trades"]
portfolios = db["portfolios"]
backtests = db["backtests"]


def save_trade(user, symbol, action, price, quantity):

    trade = {
        "user": user,
        "symbol": symbol,
        "action": action,
        "price": price,
        "quantity": quantity,
        "timestamp": datetime.utcnow()
    }

    trades.insert_one(trade)
    return trade


def get_portfolio(user):

    portfolio = portfolios.find_one({"user": user})

    if not portfolio:
        portfolio = {
            "user": user,
            "capital": 100000,
            "positions": {}
        }
        portfolios.insert_one(portfolio)

    return portfolio


def update_position(user, symbol, qty):

    portfolio = get_portfolio(user)

    positions = portfolio.get("positions", {})

    if symbol not in positions:
        positions[symbol] = 0

    positions[symbol] += qty

    portfolios.update_one(
        {"user": user},
        {"$set": {"positions": positions}}
    )