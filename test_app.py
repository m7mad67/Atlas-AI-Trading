import os
import tempfile
import unittest
from pathlib import Path

from database import place_order, portfolio_state
from market_data import ASSETS, build_market_data
from analytics import historical_risk, optimize_strategy


class AtlasPaperTradingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["ATLAS_DB_PATH"] = str(Path(self.temp_dir.name) / "test.sqlite3")
        import importlib
        import database

        importlib.reload(database)
        self.database = database
        self.database.initialize_database()

    def tearDown(self):
        self.temp_dir.cleanup()
        os.environ.pop("ATLAS_DB_PATH", None)

    def test_starts_with_virtual_balance(self):
        state = self.database.portfolio_state()
        self.assertEqual(state["cash"], 100_000.0)
        self.assertEqual(state["positions"], {})

    def test_buy_and_sell_updates_cash_and_position(self):
        self.database.place_order("BUY", "AAPL", 2, 100)
        state = self.database.portfolio_state()
        self.assertAlmostEqual(state["positions"]["AAPL"]["quantity"], 2)
        self.assertLess(state["cash"], 99_800)
        self.database.place_order("SELL", "AAPL", 1, 110)
        state = self.database.portfolio_state()
        self.assertAlmostEqual(state["positions"]["AAPL"]["quantity"], 1)
        self.assertGreater(state["cash"], 99_900)

    def test_rejects_overspending_and_overselling(self):
        with self.assertRaises(ValueError):
            self.database.place_order("BUY", "BTC-USD", 10, 100_000)
        with self.assertRaises(ValueError):
            self.database.place_order("SELL", "AAPL", 1, 100)

    def test_demo_market_and_analytics(self):
        markets = build_market_data()
        self.assertEqual(set(markets), set(ASSETS))
        self.assertTrue({"Open", "High", "Low", "Close", "Volume"}.issubset(markets["AAPL"].columns))
        self.database.place_order("BUY", "AAPL", 1, float(markets["AAPL"]["Close"].iloc[-1]))
        risk = historical_risk(self.database.portfolio_state()["positions"], markets)
        self.assertIsNotNone(risk)
        result = optimize_strategy(markets["AAPL"]["Close"], 0.7, 5, 3)
        self.assertLess(result["train_end"], result["test_start"])
        self.assertEqual(len(result["grid"]), 9)


if __name__ == "__main__":
    unittest.main()
