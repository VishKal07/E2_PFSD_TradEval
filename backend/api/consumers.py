import json
from channels.generic.websocket import AsyncWebsocketConsumer
import yfinance as yf


class MarketConsumer(AsyncWebsocketConsumer):

    async def connect(self):

        await self.accept()

    async def receive(self, text_data):

        data = json.loads(text_data)

        symbol = data["symbol"]

        ticker = yf.Ticker(symbol)

        price = ticker.history(period="1d")["Close"].iloc[-1]

        await self.send(json.dumps({
            "symbol": symbol,
            "price": float(price)
        }))