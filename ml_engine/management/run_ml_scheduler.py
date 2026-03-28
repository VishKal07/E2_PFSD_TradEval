import schedule
import time
import logging
from django.core.management.base import BaseCommand
from ml_engine.services.training_pipeline import AntiOverfittingTrainer
from trading.models import ModelRegistry
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Run ML training scheduler with anti-overfitting measures'
    
    def handle(self, *args, **options):
        logger.info("Starting ML Scheduler...")
        
        # Schedule weekly retraining for each symbol
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Configure your symbols
        
        for symbol in symbols:
            schedule.every().monday.at("00:00").do(self.train_symbol, symbol)
            schedule.every().thursday.at("12:00").do(self.evaluate_models, symbol)
        
        # Continuous monitoring
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def train_symbol(self, symbol):
        """Train new model with walk-forward validation"""
        logger.info(f"Starting training for {symbol}")
        trainer = AntiOverfittingTrainer(symbol)
        model_id, scores = trainer.run_training_pipeline()
        
        if model_id:
            logger.info(f"Training complete for {symbol}: {model_id}")
            self.check_and_deploy_best_model(symbol)
    
    def evaluate_models(self, symbol):
        """Evaluate deployed models and replace if overfitting"""
        deployed_models = ModelRegistry.objects.filter(
            symbol=symbol,
            is_deployed=True,
            is_archived=False
        )
        
        for model in deployed_models:
            # Check if model has been deployed for > 7 days
            if model.deployment_date and model.deployment_date < datetime.now() - timedelta(days=7):
                # Evaluate recent performance
                recent_performance = self.evaluate_recent_performance(model)
                
                # If performance degraded, archive and retrain
                if recent_performance['mse'] > model.out_of_sample_metrics['avg_mse'] * 1.5:
                    logger.warning(f"Model {model.model_id} showing signs of overfitting. Archiving...")
                    model.is_deployed = False
                    model.is_archived = True
                    model.save()
                    
                    # Trigger retraining
                    self.train_symbol(symbol)
    
    def evaluate_recent_performance(self, model):
        """Evaluate model on recent out-of-sample data"""
        # Implementation for live performance evaluation
        pass
    
    def check_and_deploy_best_model(self, symbol):
        """Deploy the best performing model for a symbol"""
        best_model = ModelRegistry.objects.filter(
            symbol=symbol,
            is_deployed=False,
            is_archived=False
        ).order_by('-out_of_sample_metrics__avg_mse').first()
        
        if best_model:
            # Archive current deployed model
            ModelRegistry.objects.filter(
                symbol=symbol,
                is_deployed=True
            ).update(is_deployed=False, is_archived=True)
            
            # Deploy new best model
            best_model.is_deployed = True
            best_model.deployment_date = datetime.now()
            best_model.save()
            
            logger.info(f"Deployed new model for {symbol}: {best_model.model_id}")