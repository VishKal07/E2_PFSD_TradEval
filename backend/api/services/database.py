from pymongo import MongoClient
from datetime import datetime

# MongoDB connection
client = MongoClient("mongodb://127.0.0.1:27017")

# Database name
db = client["tradeeval_db"]

# Collection name
collection = db["results"]


def save_result(result):

    result["timestamp"] = datetime.utcnow()

    collection.insert_one(result)