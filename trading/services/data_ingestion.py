import asyncio
import websockets
import json
import logging
from datetime import datetime
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from django.core.cache import cache
from ..models import MarketData
from django.db import transaction
import numpy as np

logger = logging.getLogger(__name__)

class RealTimeDataIngestion:
    def __init__(self):
        self.websocket_url = "wss://stream.data.alpaca.markets/v2/iex"
        self.api_key = None  # Set from settings
        self.batch_buffer = []
        self.batch_size = 100
        self.running = False
        
    async def connect_and_stream(self, symbols):
        """Connect to WebSocket and stream real-time data"""
        self.running = True
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        async with websockets.connect(self.websocket_url, extra_headers=headers) as websocket:
            # Subscribe to symbols
            subscribe_msg = {
                "action": "subscribe",
                "trades": symbols,
                "quotes": symbols
            }
            await websocket.send(json.dumps(subscribe_msg))
            
            # Start batch processor
            asyncio.create_task(self._process_batch())
            
            # Receive messages
            async for message in websocket:
                await self._handle_message(message)
                
    async def _handle_message(self, message):
        """Process incoming WebSocket messages"""
        data = json.loads(message)
        
        if 'trades' in data:
            for trade in data['trades']:
                market_data = {
                    'symbol': trade['S'],
                    'timestamp': datetime.fromisoformat(trade['t']),
                    'price': trade['p'],
                    'volume': trade['s'],
                    'interval': 'tick'
                }
                self.batch_buffer.append(market_data)
                
                # Cache latest price for quick access
                cache_key = f"latest_price_{trade['S']}"
                cache.set(cache_key, trade['p'], timeout=60)
                
                # Broadcast via WebSocket to frontend
                await self._broadcast_price(trade['S'], trade['p'])
                
                if len(self.batch_buffer) >= self.batch_size:
                    await self._flush_batch()
    
    async def _flush_batch(self):
        """Flush batch to MongoDB"""
        if not self.batch_buffer:
            return
            
        try:
            # Use bulk insert for efficiency
            market_data_objects = [
                MarketData(
                    symbol=item['symbol'],
                    timestamp=item['timestamp'],
                    open=item['price'],
                    high=item['price'],
                    low=item['price'],
                    close=item['price'],
                    volume=item['volume'],
                    interval='tick'
                )
                for item in self.batch_buffer
            ]
            
            await self._bulk_insert(market_data_objects)
            
            # Also update aggregated data
            await self._update_aggregates(self.batch_buffer)
            
            self.batch_buffer.clear()
            
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
    
    async def _update_aggregates(self, ticks):
        """Update 1min, 5min, 1hour aggregates in real-time"""
        from collections import defaultdict
        
        aggregates = defaultdict(lambda: {
            'open': None, 'high': -np.inf, 'low': np.inf, 
            'close': None, 'volume': 0, 'timestamp': None
        })
        
        for tick in ticks:
            minute_key = tick['timestamp'].replace(second=0, microsecond=0)
            agg = aggregates[(tick['symbol'], minute_key)]
            
            if agg['open'] is None:
                agg['open'] = tick['price']
                agg['timestamp'] = minute_key
            agg['high'] = max(agg['high'], tick['price'])
            agg['low'] = min(agg['low'], tick['price'])
            agg['close'] = tick['price']
            agg['volume'] += tick['volume']
        
        # Store aggregates
        for (symbol, timestamp), agg_data in aggregates.items():
            await MarketData.objects.aupdate_or_create(
                symbol=symbol,
                timestamp=timestamp,
                interval='1min',
                defaults=agg_data
            )
    
    async def _broadcast_price(self, symbol, price):
        """Broadcast price update to frontend via WebSocket"""
        channel_layer = get_channel_layer()
        await channel_layer.group_send(
            f"prices_{symbol}",
            {
                'type': 'price_update',
                'symbol': symbol,
                'price': price,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    async def _bulk_insert(self, objects):
        """Bulk insert to MongoDB"""
        await MarketData.objects.abulk_create(objects)
    
    async def _process_batch(self):
        """Background task to flush batches periodically"""
        while self.running:
            await asyncio.sleep(5)
            if self.batch_buffer:
                await self._flush_batch()