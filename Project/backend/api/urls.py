from django.urls import path
from .views import backtest_view, risk_view

urlpatterns = [
    path("backtest/", backtest_view),
    path("risk/", risk_view),
]