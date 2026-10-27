from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from market_data import ASSETS


DEFAULT_BALANCE = 100_000.0
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0003
DB_PATH = Path(os.environ.get("ATLAS_DB_PATH", Path(__file__).with_name("atlas_paper.db")))


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


def initialize_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS account (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                initial_balance REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL CHECK (quantity > 0),
                reference_price REAL NOT NULL CHECK (reference_price > 0),
                fill_price REAL NOT NULL CHECK (fill_price > 0),
                fees REAL NOT NULL CHECK (fees >= 0),
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
            """
        )
        connection.execute(
            "INSERT OR IGNORE INTO account (id, initial_balance) VALUES (1, ?)",
            (DEFAULT_BALANCE,),
        )


def _read_state(connection: sqlite3.Connection) -> tuple[float, dict[str, dict[str, float]]]:
    initial_balance = float(connection.execute("SELECT initial_balance FROM account WHERE id = 1").fetchone()["initial_balance"])
    trades = connection.execute("SELECT * FROM trades ORDER BY id").fetchall()
    cash = initial_balance
    lots: dict[str, list[list[float]]] = {}
    for trade in trades:
        symbol = trade["symbol"]
        quantity = float(trade["quantity"])
        fill = float(trade["fill_price"])
        fees = float(trade["fees"])
        if trade["side"] == "BUY":
            cash -= quantity * fill + fees
            lots.setdefault(symbol, []).append([quantity, quantity * fill + fees])
        else:
            cash += quantity * fill - fees
            remaining = quantity
            for lot in lots.get(symbol, []):
                if remaining <= 1e-10:
                    break
                lot_quantity = lot[0]
                closed = min(lot_quantity, remaining)
                lot[0] -= closed
                lot[1] -= lot[1] * (closed / lot_quantity) if lot_quantity else 0.0
                remaining -= closed

    positions: dict[str, dict[str, float]] = {}
    for symbol, symbol_lots in lots.items():
        quantity = sum(lot[0] for lot in symbol_lots)
        cost_basis = sum(lot[1] for lot in symbol_lots)
        if quantity > 1e-10:
            positions[symbol] = {"quantity": quantity, "cost_basis": cost_basis}
    return cash, positions


def portfolio_state(market_data: dict | None = None) -> dict:
    with _connect() as connection:
        cash, positions = _read_state(connection)
    return {"cash": cash, "positions": positions}


def place_order(side: str, symbol: str, quantity: float, reference_price: float) -> dict:
    side = side.upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError("Choose BUY or SELL.")
    if symbol not in ASSETS:
        raise ValueError("That instrument is not in the demo market universe.")
    if not quantity or quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")
    if not reference_price or reference_price <= 0:
        raise ValueError("The demo reference price must be greater than zero.")

    fill_price = reference_price * (1 + SLIPPAGE_RATE if side == "BUY" else 1 - SLIPPAGE_RATE)
    fees = quantity * fill_price * FEE_RATE
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        cash, positions = _read_state(connection)
        if side == "BUY" and quantity * fill_price + fees > cash:
            raise ValueError(f"Insufficient virtual cash. Available: ${cash:,.2f}.")
        if side == "SELL" and quantity > positions.get(symbol, {}).get("quantity", 0.0) + 1e-10:
            owned = positions.get(symbol, {}).get("quantity", 0.0)
            raise ValueError(f"Sell quantity exceeds your position. Available to sell: {owned:.6f}.")
        cursor = connection.execute(
            """
            INSERT INTO trades (side, symbol, quantity, reference_price, fill_price, fees)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (side, symbol, quantity, reference_price, fill_price, fees),
        )
        order_id = cursor.lastrowid
    return {
        "id": order_id,
        "side": side,
        "symbol": symbol,
        "quantity": quantity,
        "reference_price": reference_price,
        "fill_price": fill_price,
        "fees": fees,
    }


def get_trades(limit: int = 100) -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 500)),),
        ).fetchall()
    return [dict(row) for row in rows]
