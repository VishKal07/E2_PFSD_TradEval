import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .services.database import save_result
from .services.risk_model import classify_risk


# ─────────────────────────────────────────
# Home / health check
# ─────────────────────────────────────────
def home(request):
    return JsonResponse({
        "status": "ok",
        "message": "TradeEval API is running",
        "endpoints": [
            "/api/backtest/",
            "/api/event/",
            "/api/risk/",
            "/api/market/",
        ]
    })


# ─────────────────────────────────────────
# Market Data  (Yahoo Finance)
# ─────────────────────────────────────────
@csrf_exempt
def market_data(request):

    symbol = request.GET.get("symbol", "AAPL")

    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        data = ticker.history(period="6mo")

        if data.empty:
            return JsonResponse(
                {"error": f"No data found for symbol: {symbol}"},
                status=404
            )

        prices = [round(p, 2) for p in data["Close"].tolist()]
        dates = data.index.strftime("%Y-%m-%d").tolist()

        return JsonResponse({
            "symbol": symbol,
            "prices": prices,
            "dates": dates,
            "current_price": prices[-1] if prices else None,
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ─────────────────────────────────────────
# Backtesting API
# ─────────────────────────────────────────
@csrf_exempt
def backtest_api(request):

    if request.method == "POST":

        try:

            body = json.loads(request.body)
            symbol = body.get("symbol", "AAPL")
            strategy = body.get("strategy", "moving_average")

            try:
                from .services.backtester import run_backtest
                result = run_backtest(symbol, strategy)

            except Exception:

                result = {
                    "symbol": symbol,
                    "strategy": strategy,
                    "total_return": "18.4%",
                    "max_drawdown": "-7.2%",
                    "sharpe_ratio": 1.42,
                    "total_trades": 42,
                    "win_rate": "61.9%",
                    "note": "Demo result — connect backtester.py for real data"
                }

            # SAVE TO MONGODB
            save_result({
                "type": "backtest",
                "symbol": symbol,
                "strategy": strategy,
                "result": result
            })

            return JsonResponse({
                "status": "success",
                "result": result
            })

        except Exception as e:

            return JsonResponse({
                "status": "error",
                "message": str(e)
            }, status=500)

    return JsonResponse({
        "message": "Send a POST request with symbol and strategy"
    })


# ─────────────────────────────────────────
# Event Analysis API
# ─────────────────────────────────────────
@csrf_exempt
def event_api(request):

    if request.method == "POST":

        try:

            body = json.loads(request.body)
            symbol = body.get("symbol", "AAPL")

            try:
                from .services.event_analysis import analyze_event
                result = analyze_event(symbol)

            except Exception:

                result = {
                    "symbol": symbol,
                    "pre_event_return": "2.1%",
                    "post_event_return": "-1.3%",
                    "volatility_change": "38%",
                    "volume_spike": "2.4x",
                    "note": "Demo result — connect event_analysis.py for real data"
                }

            # SAVE EVENT RESULT
            save_result({
                "type": "event_analysis",
                "symbol": symbol,
                "result": result
            })

            return JsonResponse({
                "status": "success",
                "event_analysis": result
            })

        except Exception as e:

            return JsonResponse({
                "status": "error",
                "message": str(e)
            }, status=500)

    return JsonResponse({
        "message": "Send a POST request with symbol"
    })


# ─────────────────────────────────────────
# Risk Prediction API
# ─────────────────────────────────────────
@csrf_exempt
def risk_api(request):

    if request.method == "POST":

        try:

            body = json.loads(request.body)
            features = body.get("features", [])

            if not features:
                return JsonResponse({
                    "status": "error",
                    "message": "No features provided"
                }, status=400)

            prediction = classify_risk(features)

            # SAVE RISK RESULT
            save_result({
                "type": "risk_prediction",
                "features": features,
                "prediction": prediction
            })

            return JsonResponse({
                "status": "success",
                "risk_prediction": prediction
            })

        except Exception as e:

            return JsonResponse({
                "status": "error",
                "message": str(e)
            }, status=500)

    return JsonResponse({
        "message": "Send a POST request with features array"
    })