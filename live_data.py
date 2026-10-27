"""Optional historical quotes; no paid keys, no automatic orders."""
from __future__ import annotations
import pandas as pd


def normalize_history(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError('No historical bars returned')
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(name in frame for name in needed):
        raise ValueError('Missing OHLCV fields')
    frame = frame[needed].copy()
    for col in needed:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame = frame.replace([float('inf'), float('-inf')], float('nan')).dropna(subset=needed)
    frame = frame[(frame['Close'] > 0) & (frame['High'] >= frame['Low'])]
    frame.index = pd.to_datetime(frame.index, utc=True).tz_localize(None).normalize()
    frame = frame[~frame.index.duplicated(keep='last')].sort_index()
    if len(frame) < 150:
        raise ValueError('Too few valid historical bars (need at least 150)')
    return frame.rename_axis('Date')


def fetch_historical_data(symbols: list[str], period: str = '2y') -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Returns only verified, nonempty historical data; failures are not silently replaced."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError('Install yfinance from requirements.txt') from exc
    result, failures = {}, {}
    for symbol in symbols:
        try:
            raw = yf.download(symbol, period=period, interval='1d', auto_adjust=True,
                              progress=False, threads=False, timeout=12)
            result[symbol] = normalize_history(raw)
        except Exception as exc:
            failures[symbol] = str(exc)[:180]
    return result, failures
