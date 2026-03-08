import json
from django.http import JsonResponse
from .services.strategy_engine import run_strategy
from .services.risk_api import classify_risk

def backtest_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=400)

    data = json.loads(request.body)

    result = run_strategy(
        symbol=data["symbol"],
        strategy=data["strategy"]
    )
    return JsonResponse(result)

def risk_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=400)

    metrics = json.loads(request.body)
    risk = classify_risk(metrics)

    return JsonResponse({"risk_level": risk})