from django.urls import path
from . import views

urlpatterns = [

    path("market/", views.market_data),
    path("backtest/", views.backtest),
    path("ai/", views.ai_analysis),

]