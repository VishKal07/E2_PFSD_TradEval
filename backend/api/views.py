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
    # ─────────────────────────────────────────────────────────────────
# Combined Analysis API  — event + risk in one call
# ─────────────────────────────────────────────────────────────────
@csrf_exempt
def analyze_api(request):

    if request.method != "POST":
        return JsonResponse({
            "message": "Send a POST request with symbol and strategy"
        })

    try:
        body     = json.loads(request.body)
        symbol   = body.get("symbol",   "AAPL")
        strategy = body.get("strategy", "moving_average")

        # ── Step 1: event analysis + news sentiment ───────────────
        from .services.event_analysis import analyze_event
        event_result = analyze_event(symbol)

        if "error" in event_result:
            return JsonResponse({
                "status": "error",
                "message": event_result["error"]
            }, status=400)

        # ── Step 2: backtest the strategy ─────────────────────────
        from .services.backtester import run_backtest
        backtest_result = run_backtest(symbol, strategy)

        # ── Step 3: build risk features from what we already have ─
        # These match FEATURE_NAMES in risk_model.py exactly
        import yfinance as yf
        import numpy as np

        try:
            ticker = yf.Ticker(symbol)
            hist   = ticker.history(period="3mo")
            prices = hist["Close"].tolist()
            r      = np.diff(prices) / prices[:-1]

            return_1d  = round(float(r[-1]),          6) if len(r) >= 1  else 0.0
            return_5d  = round(float(np.mean(r[-5:])), 6) if len(r) >= 5  else 0.0
            return_20d = round(float(np.mean(r[-20:])),6) if len(r) >= 20 else 0.0
            volatility = round(float(np.std(r)),       6)

            features = [return_1d, return_5d, return_20d, volatility]

        except Exception:
            features = [0.0, 0.0, 0.0, 0.0]

        # ── Step 4: risk classification ───────────────────────────
        from .services.risk_model import classify_risk
        risk_result = classify_risk(features)

        # ── Step 5: build final combined response ─────────────────
        combined = {
            "symbol":   symbol,
            "strategy": strategy,

            # behavior analysis
            "behavior_summary":    event_result.get("behavior_summary"),
            "news_signal":         event_result.get("news_signal"),
            "volatility_pct":      event_result.get("volatility_pct"),
            "recent_return_pct":   event_result.get("recent_return_pct"),
            "avg_sentiment_score": event_result.get("avg_sentiment_score"),
            "news_breakdown":      event_result.get("news_breakdown", {}),
            "top_news": event_result.get("news", [])[:3],

            # backtest
            "backtest": backtest_result,

            # risk
            "risk_label":    risk_result.get("risk_label"),
            "risk_level":    risk_result.get("risk_level"),
            "confidence":    risk_result.get("confidence"),
            "probabilities": risk_result.get("probabilities", {}),

            # earnings
            "earnings_calendar": event_result.get("earnings_calendar", {}),
        }

        # ── Step 6: save to MongoDB ───────────────────────────────
        save_result({
            "type":     "full_analysis",
            "symbol":   symbol,
            "strategy": strategy,
            "result":   combined,
        })

        return JsonResponse({
            "status": "success",
            "analysis": combined
        })

    except Exception as e:
        return JsonResponse({
            "status":  "error",
            "message": str(e)
        }, status=500)
