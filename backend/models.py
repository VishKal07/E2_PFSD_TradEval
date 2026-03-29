from django.db import models

class MarketData(models.Model):
    id = models.AutoField(primary_key=True)
    symbol = models.CharField(max_length=20)
    timestamp = models.DateTimeField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.FloatField()
    timeframe = models.CharField(max_length=10, default='1m')
    source = models.CharField(max_length=50, default='binance')
    
    class Meta:
        indexes = [
            models.Index(fields=['symbol', 'timestamp']),
        ]
    
    def __str__(self):
        return f"{self.symbol} - {self.timestamp}"

class Prediction(models.Model):
    id = models.AutoField(primary_key=True)
    symbol = models.CharField(max_length=20)
    timestamp = models.DateTimeField(auto_now_add=True)
    current_price = models.FloatField()
    predicted_price = models.FloatField()
    predicted_return = models.FloatField()
    confidence = models.FloatField()
    model_version = models.CharField(max_length=50)
    
    class Meta:
        indexes = [
            models.Index(fields=['symbol', 'timestamp']),
        ]

class TrainedModel(models.Model):
    id = models.AutoField(primary_key=True)
    version = models.CharField(max_length=50, unique=True)
    symbol = models.CharField(max_length=20)
    model_path = models.CharField(max_length=500)
    train_accuracy = models.FloatField(null=True, blank=True)
    test_accuracy = models.FloatField(null=True, blank=True)
    trained_on = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)
    data_points = models.IntegerField()
    
    def __str__(self):
        return f"Model {self.version} - {self.symbol}"