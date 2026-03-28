from djongo import models
from djongo.models import JSONField
from datetime import datetime
import uuid

class MarketData(models.Model):
    """Time-series market data collection"""
    _id = models.ObjectIdField()
    symbol = models.CharField(max_length=20)
    timestamp = models.DateTimeField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.IntegerField()
    interval = models.CharField(max_length=10, choices=[
        ('tick', 'Tick'),
        ('1min', '1 Minute'),
        ('5min', '5 Minutes'),
        ('15min', '15 Minutes'),
        ('1hour', '1 Hour'),
        ('1day', '1 Day')
    ])
    metadata = JSONField(default=dict)
    
    class Meta:
        indexes = [
            models.Index(fields=['symbol', 'timestamp']),
            models.Index(fields=['interval']),
        ]
        unique_together = [['symbol', 'timestamp', 'interval']]

class Trade(models.Model):
    """Trade execution records"""
    _id = models.ObjectIdField()
    user_id = models.CharField(max_length=100)
    symbol = models.CharField(max_length=20)
    side = models.CharField(max_length=10, choices=[('buy', 'Buy'), ('sell', 'Sell')])
    quantity = models.FloatField()
    price = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='pending')
    order_type = models.CharField(max_length=20, default='market')
    stop_loss = models.FloatField(null=True, blank=True)
    take_profit = models.FloatField(null=True, blank=True)
    model_prediction_id = models.CharField(max_length=100, null=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['user_id', 'timestamp']),
            models.Index(fields=['symbol', 'status']),
        ]

class ModelRegistry(models.Model):
    """Track ML models to prevent overfitting"""
    _id = models.ObjectIdField()
    model_id = models.CharField(max_length=100, unique=True)
    model_type = models.CharField(max_length=50)
    version = models.IntegerField()
    trained_at = models.DateTimeField(auto_now_add=True)
    train_start_date = models.DateTimeField()
    train_end_date = models.DateTimeField()
    features_used = models.JSONField()
    hyperparameters = models.JSONField()
    performance_metrics = models.JSONField()
    out_of_sample_metrics = models.JSONField()
    is_deployed = models.BooleanField(default=False)
    deployment_date = models.DateTimeField(null=True)
    is_archived = models.BooleanField(default=False)
    
    class Meta:
        indexes = [
            models.Index(fields=['model_id', 'is_deployed']),
            models.Index(fields=['trained_at']),
        ]

class Prediction(models.Model):
    """Model predictions with confidence"""
    _id = models.ObjectIdField()
    model_id = models.CharField(max_length=100)
    symbol = models.CharField(max_length=20)
    timestamp = models.DateTimeField()
    predicted_price = models.FloatField()
    confidence = models.FloatField()
    direction = models.CharField(max_length=10, choices=[('up', 'Up'), ('down', 'Down')])
    features_at_prediction = models.JSONField()
    actual_outcome = models.FloatField(null=True)
    actual_timestamp = models.DateTimeField(null=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['model_id', 'timestamp']),
            models.Index(fields=['symbol', 'timestamp']),
        ]

class UserPortfolio(models.Model):
    """User portfolio embedded document"""
    user_id = models.CharField(max_length=100, primary_key=True)
    cash_balance = models.FloatField(default=100000)
    holdings = models.JSONField(default=dict)
    portfolio_value = models.FloatField(default=100000)
    last_updated = models.DateTimeField(auto_now=True)
    risk_profile = models.JSONField(default=dict)
    trading_history = models.JSONField(default=list)
    
    class Meta:
        indexes = [
            models.Index(fields=['user_id']),
        ]