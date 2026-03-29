import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
import logging

logger = logging.getLogger(__name__)

class TradingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.symbol = self.scope['url_route']['kwargs'].get('symbol', 'BTCUSDT')
        self.room_group_name = f'trading_{self.symbol}'
        
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()
        
        from trading.services.data_ingestion import RealTimeDataIngestion
        self.ingestion = RealTimeDataIngestion()
        self.data_task = asyncio.create_task(self.stream_realtime_data())
        logger.info(f"WebSocket connected for {self.symbol}")
    
    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
        if hasattr(self, 'data_task'):
            self.data_task.cancel()
        if hasattr(self, 'ingestion'):
            await self.ingestion.close()
        logger.info(f"WebSocket disconnected for {self.symbol}")
    
    async def stream_realtime_data(self):
        try:
            while True:
                try:
                    data = await self.ingestion.fetch_realtime_data(self.symbol, limit=1)
                    
                    if data:
                        # Convert datetime to string for JSON
                        for item in data:
                            if 'timestamp' in item:
                                item['timestamp'] = item['timestamp'].isoformat()
                        await self.send(text_data=json.dumps(data[0]))
                    
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    await asyncio.sleep(10)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Stream task error: {e}")
    
    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            if data.get('type') == 'subscribe':
                self.symbol = data.get('symbol', 'BTCUSDT')
                logger.info(f"Subscribed to {self.symbol}")
        except Exception as e:
            logger.error(f"Error receiving message: {e}")