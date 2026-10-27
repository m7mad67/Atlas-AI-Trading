from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


ASSETS: dict[str, dict[str, str | float]] = {
    "AAPL": {"name": "Apple", "kind": "Stock", "start": 172.0, "drift": 0.00025, "vol": 0.018, "seed": 11},
    "MSFT": {"name": "Microsoft", "kind": "Stock", "start": 365.0, "drift": 0.00032, "vol": 0.016, "seed": 17},
    "NVDA": {"name": "NVIDIA", "kind": "Stock", "start": 118.0, "drift": 0.00055, "vol": 0.031, "seed": 23},
    "AMZN": {"name": "Amazon", "kind": "Stock", "start": 188.0, "drift": 0.00028, "vol": 0.021, "seed": 31},
    "TSLA": {"name": "Tesla", "kind": "Stock", "start": 244.0, "drift": 0.00012, "vol": 0.035, "seed": 41},
    "BTC-USD": {"name": "Bitcoin", "kind": "Crypto", "start": 63500.0, "drift": 0.0005, "vol": 0.038, "seed": 53},
    "ETH-USD": {"name": "Ethereum", "kind": "Crypto", "start": 3250.0, "drift": 0.00045, "vol": 0.043, "seed": 67},
    "SOL-USD": {"name": "Solana", "kind": "Crypto", "start": 148.0, "drift": 0.00065, "vol": 0.055, "seed": 79},
}


def _asset_frame(symbol: str, asset: dict[str, str | float], sessions: int = 520) -> pd.DataFrame:
    rng = np.random.default_rng(int(asset["seed"]))
    dates = pd.bdate_range(end=pd.Timestamp.now(tz="UTC").normalize().tz_localize(None), periods=sessions)
    base = float(asset["start"])
    drift = float(asset["drift"])
    volatility = float(asset["vol"])
    regimes = 0.65 + 0.35 * np.sin(np.linspace(0, 8 * np.pi, sessions - 1)) ** 2
    returns = rng.normal(drift, volatility * regimes, size=sessions - 1)
    close = np.concatenate(([base], base * np.cumprod(1 + returns)))
    overnight = rng.normal(0, volatility * 0.25, size=sessions)
    open_prices = close * (1 + overnight)
    ranges = np.abs(rng.normal(volatility * 0.65, volatility * 0.25, size=sessions))
    high = np.maximum(open_prices, close) * (1 + ranges)
    low = np.minimum(open_prices, close) * np.maximum(0.01, (1 - ranges))
    volume_scale = 2_500_000 if asset["kind"] == "Stock" else 950
    volume = rng.lognormal(np.log(volume_scale), 0.48, size=sessions).astype(np.int64)
    return pd.DataFrame(
        {"Open": open_prices, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    ).rename_axis("Date")


def build_market_data() -> dict[str, pd.DataFrame]:
    """Return deterministic synthetic daily bars; no external service is called."""
    return {symbol: _asset_frame(symbol, asset) for symbol, asset in ASSETS.items()}


def market_timestamp(market_data: dict[str, pd.DataFrame]) -> str:
    last_date = max(frame.index[-1] for frame in market_data.values())
    return f"{last_date.strftime('%Y-%m-%d')} (demo series)"
