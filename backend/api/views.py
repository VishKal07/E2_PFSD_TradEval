from django.http import JsonResponse
from .services.market_data import get_stock_data
from .services.backtester import run_backtest
from .services.claude_ai import ask_claude


def market_data(request):

    symbol = request.GET.get("symbol", "AAPL")

    data = get_stock_data(symbol)

    return JsonResponse(data)


def backtest(request):

    symbol = request.GET.get("symbol", "AAPL")

    result = run_backtest(symbol)

    return JsonResponse(result)


def ai_analysis(request):

    question = request.GET.get("q", "Analyze this stock")

    analysis = ask_claude(question)

    return JsonResponse({
        "analysis": analysis
    })