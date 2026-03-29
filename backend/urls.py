from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'market-data', views.MarketDataViewSet)
router.register(r'predictions', views.PredictionViewSet)
router.register(r'models', views.TrainedModelViewSet)

urlpatterns = [
    # API endpoints
    path('api/', include(router.urls)),
    path('api/realtime/', views.realtime_market_data, name='realtime_data'),
    path('api/market/', views.get_market_data, name='market_data'),
    path('api/backtest/', views.run_backtest, name='backtest'),
    path('api/risk/', views.predict_risk, name='risk_prediction'),
    path('api/predict/', views.predict_price, name='predict_price'),
    path('api/train/', views.trigger_training, name='train_model'),
    
    # Frontend routes
    path('', views.serve_landing, name='home'),           # Landing page at root
    path('simulator/', views.serve_trading, name='trading'),  # Trading simulator
    path('trading/', views.serve_trading, name='trading_alt'), # Alternative URL
]