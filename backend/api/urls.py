from django.urls import path
from .views import home, market_data, backtest_api, event_api, risk_api

urlpatterns = [
    path("",         home),
    path("market/",  market_data),
    path("backtest/",backtest_api),
    path("event/",   event_api),
    path("risk/",    risk_api),
]
