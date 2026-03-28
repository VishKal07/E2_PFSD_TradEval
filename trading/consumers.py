import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.core.cache import cache
import logging

logger = logging.getLogger(__name__)

class PriceConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.symbol = self.scope['url_route']['kwargs']['symbol']
        self.group_name = f'prices_{self.symbol}'
        
        # Join group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        
        await self.accept()
        
        # Send latest cached price
        latest_price = await self.get_cached_price(self.symbol)
        if latest_price:
            await self.send(text_data=json.dumps({
                'type': 'initial_price',
                'symbol': self.symbol,
                'price': latest_price
            }))
    
    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )
    
    async def receive(self, text_data):
        """Handle messages from client"""
        data = json.loads(text_data)
        
        if data.get('type') == 'subscribe_prediction':
            # Subscribe to model predictions for this symbol
            await self.channel_layer.group_add(
                f'predictions_{self.symbol}',
                self.channel_name
            )
    
    async def price_update(self, event):
        """Send price update to WebSocket"""
        await self.send(text_data=json.dumps({
            'type': 'price_update',
            'symbol': event['symbol'],
            'price': event['price'],
            'timestamp': event['timestamp']
        }))
    
    async def prediction_update(self, event):
        """Send prediction update to WebSocket"""
        await self.send(text_data=json.dumps({
            'type': 'prediction',
            'symbol': event['symbol'],
            'predicted_price': event['predicted_price'],
            'confidence': event['confidence'],
            'direction': event['direction']
        }))
    
    @database_sync_to_async
    def get_cached_price(self, symbol):
        return cache.get(f"latest_price_{symbol}")