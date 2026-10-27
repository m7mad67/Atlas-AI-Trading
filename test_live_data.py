import unittest
import pandas as pd
from live_data import normalize_history

class HistoricalDataTests(unittest.TestCase):
    def test_valid_history(self):
        idx = pd.date_range('2025-01-01', periods=200, freq='D', tz='UTC')
        frame = pd.DataFrame({'Open': 10., 'High': 12., 'Low': 9., 'Close': 11., 'Volume': 100}, index=idx)
        result = normalize_history(frame)
        self.assertEqual(len(result), 200)
        self.assertIsNone(result.index.tz)

    def test_reject_short_history(self):
        with self.assertRaises(ValueError):
            normalize_history(pd.DataFrame({'Close': [1.]}))

if __name__ == '__main__':
    unittest.main()
