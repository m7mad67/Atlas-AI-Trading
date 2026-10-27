from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from market_data import ASSETS


def portfolio_summary(cash: float, positions: dict[str, dict[str, float]], market_data: dict[str, pd.DataFrame]) -> dict[str, float]:
    market_value = sum(
        position["quantity"] * float(market_data[symbol]["Close"].iloc[-1])
        for symbol, position in positions.items()
    )
    cost_basis = sum(position["cost_basis"] for position in positions.values())
    equity = cash + market_value
    return {
        "cash": cash,
        "market_value": market_value,
        "equity": equity,
        "cost_basis": cost_basis,
        "unrealized_pnl": market_value - cost_basis,
        "total_pnl": equity - 100_000.0,
    }


def portfolio_positions(positions: dict[str, dict[str, float]], market_data: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    total_market_value = sum(
        holding["quantity"] * float(market_data[symbol]["Close"].iloc[-1])
        for symbol, holding in positions.items()
    )
    for symbol, holding in positions.items():
        price = float(market_data[symbol]["Close"].iloc[-1])
        market_value = holding["quantity"] * price
        rows.append(
            {
                "Symbol": symbol,
                "Quantity": round(holding["quantity"], 6),
                "Avg. cost": round(holding["cost_basis"] / holding["quantity"], 2),
                "Last demo close": round(price, 2),
                "Market value": round(market_value, 2),
                "Unrealized P&L": round(market_value - holding["cost_basis"], 2),
                "Weight": market_value / total_market_value if total_market_value else 0.0,
            }
        )
    return rows


def _portfolio_returns(positions: dict[str, dict[str, float]], market_data: dict[str, pd.DataFrame]) -> pd.Series | None:
    if not positions:
        return None
    symbols = list(positions)
    weights = {}
    for symbol in symbols:
        market_value = positions[symbol]["quantity"] * float(market_data[symbol]["Close"].iloc[-1])
        weights[symbol] = market_value
    total = sum(weights.values())
    if total <= 0:
        return None
    price_returns = pd.concat(
        [market_data[symbol]["Close"].pct_change().rename(symbol) for symbol in symbols],
        axis=1,
        join="inner",
    ).dropna()
    weight_series = pd.Series({symbol: value / total for symbol, value in weights.items()})
    return (price_returns * weight_series).sum(axis=1)


def historical_risk(positions: dict[str, dict[str, float]], market_data: dict[str, pd.DataFrame]) -> dict | None:
    returns = _portfolio_returns(positions, market_data)
    if returns is None or returns.empty:
        return None
    growth = (1 + returns).cumprod()
    drawdown = growth / growth.cummax() - 1
    return {
        "volatility": float(returns.std(ddof=1) * np.sqrt(252)),
        "max_drawdown": float(drawdown.min()),
        "series": growth,
    }


def rolling_stress(positions: dict[str, dict[str, float]], market_data: dict[str, pd.DataFrame]) -> list[dict]:
    returns = _portfolio_returns(positions, market_data)
    if returns is None:
        return []
    scenarios = []
    for window in (1, 5, 20):
        window_return = (1 + returns).rolling(window).apply(np.prod, raw=True) - 1
        if window_return.dropna().empty:
            continue
        worst_date = window_return.idxmin()
        scenarios.append(
            {
                "Window": f"{window} session" if window == 1 else f"{window} sessions",
                "Worst demo return": f"{window_return.loc[worst_date]:.2%}",
                "Window end": worst_date.strftime("%Y-%m-%d"),
                "Based on": "Synthetic history",
            }
        )
    return scenarios


def _features(close: pd.Series) -> pd.DataFrame:
    returns = close.pct_change()
    return pd.DataFrame(
        {
            "momentum_5": close.pct_change(5),
            "momentum_20": close.pct_change(20),
            "momentum_60": close.pct_change(60),
            "volatility_20": returns.rolling(20).std(),
            "ma_gap_20": close / close.rolling(20).mean() - 1,
            "ma_gap_50": close / close.rolling(50).mean() - 1,
        },
        index=close.index,
    )


def _train_signal(close: pd.Series) -> dict[str, float | str]:
    features = _features(close)
    future_return = close.shift(-5) / close - 1
    labels = pd.Series(np.where(future_return > 0.005, "BUY", np.where(future_return < -0.005, "SELL", "HOLD")), index=close.index)
    dataset = features.assign(target=labels).dropna().iloc[:-5]
    feature_columns = list(features.columns)
    split = int(len(dataset) * 0.8)
    train, test = dataset.iloc[:split], dataset.iloc[split:]
    model = RandomForestClassifier(
        n_estimators=70,
        max_depth=4,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=19,
        n_jobs=1,
    )
    model.fit(train[feature_columns], train["target"])
    accuracy = float((model.predict(test[feature_columns]) == test["target"]).mean()) if len(test) else 0.0
    current_features = features.iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0)
    probabilities = model.predict_proba(current_features)[0]
    result: dict[str, float | str] = {
        "Signal": str(model.predict(current_features)[0]),
        "Holdout accuracy": accuracy,
    }
    for label in ("BUY", "HOLD", "SELL"):
        result[f"P({label})"] = float(probabilities[list(model.classes_).index(label)]) if label in model.classes_ else 0.0
    return result


def build_signal_table(market_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for symbol, frame in market_data.items():
        result = _train_signal(frame["Close"])
        rows.append(
            {
                "Symbol": symbol,
                "Name": ASSETS[symbol]["name"],
                "Research signal": result["Signal"],
                "P(BUY)": result["P(BUY)"],
                "P(HOLD)": result["P(HOLD)"],
                "P(SELL)": result["P(SELL)"],
                "Holdout accuracy": result["Holdout accuracy"],
            }
        )
    table = pd.DataFrame(rows)
    for column in ("P(BUY)", "P(HOLD)", "P(SELL)", "Holdout accuracy"):
        table[column] = table[column].map(lambda value: f"{value:.1%}")
    return table


def strategy_returns(close: pd.Series, short: int, long: int, fee_bps: float, slippage_bps: float) -> pd.Series:
    short_average = close.rolling(short, min_periods=short).mean()
    long_average = close.rolling(long, min_periods=long).mean()
    signal = (short_average > long_average).astype(float).where(long_average.notna(), 0.0)
    position = signal.shift(1).fillna(0.0)
    turnover = position.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(position.iloc[0])
    cost_per_turnover = (fee_bps + slippage_bps) / 10_000
    daily_returns = close.pct_change().fillna(0.0)
    return position * daily_returns - turnover.fillna(0.0) * cost_per_turnover


def _metrics(returns: pd.Series) -> tuple[float, float, float, int]:
    if returns.empty:
        return 0.0, 0.0, 0.0, 0
    equity = (1 + returns).cumprod()
    annual_vol = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
    drawdown = equity / equity.cummax() - 1
    return float(equity.iloc[-1] - 1), annual_vol, float(drawdown.min()), int((returns.abs() > 0).sum())


def optimize_strategy(close: pd.Series, split_ratio: float, fee_bps: float, slippage_bps: float) -> dict:
    close = close.dropna()
    split = int(len(close) * split_ratio)
    training = close.iloc[:split]
    parameter_grid = [(short, long) for short in (5, 10, 20) for long in (30, 50, 100) if short < long]
    grid = []
    for short, long in parameter_grid:
        returns = strategy_returns(training, short, long, fee_bps, slippage_bps)
        net_return, volatility, drawdown, trades = _metrics(returns)
        sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 and returns.std(ddof=1) else 0.0
        grid.append({"Short": short, "Long": long, "Training return": net_return, "Training volatility": volatility, "Training max drawdown": drawdown, "Training Sharpe": sharpe, "Training active sessions": trades})
    winner = max(grid, key=lambda item: item["Training Sharpe"])
    short, long = int(winner["Short"]), int(winner["Long"])
    all_returns = strategy_returns(close, short, long, fee_bps, slippage_bps)
    test_returns = all_returns.iloc[split:]
    test_return, test_volatility, test_drawdown, test_trades = _metrics(test_returns)
    test_equity = (1 + test_returns).cumprod()
    benchmark = (1 + close.pct_change().fillna(0).iloc[split:]).cumprod()
    return {
        "short": short,
        "long": long,
        "train_start": training.index[0].strftime("%Y-%m-%d"),
        "train_end": training.index[-1].strftime("%Y-%m-%d"),
        "test_start": close.index[split].strftime("%Y-%m-%d"),
        "test_end": close.index[-1].strftime("%Y-%m-%d"),
        "test_return": test_return,
        "test_volatility": test_volatility,
        "test_drawdown": test_drawdown,
        "test_trades": test_trades,
        "equity": test_equity,
        "benchmark": benchmark,
        "grid": grid,
    }
