# ── Feature engineering ────────────────────────────────────────────
def compute_features(g):
    g = g.copy()
    r = g["close"].pct_change()

    # volatility — used ONLY for labeling, not as a feature
    g["volatility"] = r.rolling(WINDOW, min_periods=5).std() * np.sqrt(252)

    # max drawdown — independent of return direction
    def dd(x):
        c = (1 + x).cumprod()
        return ((c - c.cummax()) / c.cummax()).min()
    g["max_drawdown"] = r.rolling(WINDOW, min_periods=5).apply(dd, raw=False)

    # volume ratio — is volume unusually high?
    if has_volume:
        avg_vol = g["volume"].rolling(WINDOW, min_periods=5).mean()
        g["volume_ratio"] = g["volume"] / avg_vol.replace(0, np.nan)
    else:
        g["volume_ratio"] = 1.0

    # high-low range as % of close — measures intraday volatility
    if has_high_low:
        g["high_low_range"] = (g["high"] - g["low"]) / g["close"].replace(0, np.nan)
        g["high_low_range"] = g["high_low_range"].rolling(WINDOW, min_periods=5).mean()
        # where did close land between high and low? (0=at low, 1=at high)
        hl_diff = (g["high"] - g["low"]).replace(0, np.nan)
        g["close_position"] = ((g["close"] - g["low"]) / hl_diff).rolling(WINDOW, min_periods=5).mean()
    else:
        g["high_low_range"] = 0.02   # neutral fallback
        g["close_position"] = 0.5

    return g

print("Engineering features...")
data = data.groupby("symbol", group_keys=False).apply(compute_features)
data = data.reset_index(drop=True)  # keep symbol as a column

# ❌ Removed this block:
# if "symbol" not in data.columns:
#     data["symbol"] = data.index.get_level_values("symbol")

data.dropna(subset=FEATURES + ["volatility"], inplace=True)
print(f"Rows after feature engineering: {len(data)}")
print(f"Columns after engineering: {list(data.columns)}")  # debug check
