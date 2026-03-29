import ccxt.async_support as ccxt
import asyncio
from datetime import datetime
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class RealTimeDataIngestion:
    def __init__(self):
        self.exchange = None
        self._init_exchange()
    
    def _init_exchange(self):
        try:
            self.exchange = ccxt.binance({
                'rateLimit': 1200,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
                'timeout': 30000,
            })
            logger.info("Binance exchange initialized")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    async def fetch_realtime_data(self, symbol: str, limit: int = 100) -> List[Dict]:
        if not self.exchange:
            raise Exception("Exchange not initialized")
        
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, '1m', limit=limit)
            
            if not ohlcv:
                raise Exception(f"No data returned for {symbol}")
            
            data = []
            for candle in ohlcv:
                data.append({
                    'symbol': symbol,
                    'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5],
                    'timeframe': '1m',
                    'source': 'binance'
                })
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            raise
    
    async def fetch_historical_data(self, symbol: str, since: datetime, until: datetime = None) -> List[Dict]:
        if not self.exchange:
            raise Exception("Exchange not initialized")
        
        try:
            since_ts = int(since.timestamp() * 1000)
            until_ts = int(until.timestamp() * 1000) if until else None
            
            all_candles = []
            current_since = since_ts
            batch_count = 0
            max_batches = 100
            
            while (until_ts is None or current_since < until_ts) and batch_count < max_batches:
                candles = await self.exchange.fetch_ohlcv(
                    symbol, '1h', since=current_since, limit=1000
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                current_since = candles[-1][0] + 1
                batch_count += 1
                await asyncio.sleep(self.exchange.rateLimit / 1000)
            
            logger.info(f"Fetched {len(all_candles)} historical candles for {symbol}")
            
            data = []
            for candle in all_candles:
                data.append({
                    'symbol': symbol,
                    'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5],
                    'timeframe': '1h',
                    'source': 'binance'
                })
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    async def close(self):
        if self.exchange:
            await self.exchange.close()
            logger.info("Exchange connection closed")