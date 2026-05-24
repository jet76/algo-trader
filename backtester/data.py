from typing import List

import numpy as np
import pandas as pd
import yfinance as yf


class DataFeed:
    def __init__(
        self,
        symbols: List[str],
        start: str,
        end: str,
        interval: str = "1d",
        mock: bool = False,
    ):
        self.symbols = symbols
        self.data: dict[str, pd.DataFrame] = {}
        if mock:
            self._load_mock(start, end)
        else:
            self._load(start, end, interval)

    def _load(self, start: str, end: str, interval: str) -> None:
        for symbol in self.symbols:
            df = yf.Ticker(symbol).history(start=start, end=end, interval=interval)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            self.data[symbol] = df

    def _load_mock(self, start: str, end: str) -> None:
        """Generate synthetic random-walk OHLCV data for testing without network access."""
        dates = pd.bdate_range(start=start, end=end)
        rng = np.random.default_rng(seed=42)
        for symbol in self.symbols:
            n = len(dates)
            close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
            high = close * (1 + rng.uniform(0, 0.02, n))
            low = close * (1 - rng.uniform(0, 0.02, n))
            open_ = low + rng.uniform(0, 1, n) * (high - low)
            volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
            self.data[symbol] = pd.DataFrame(
                {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
                index=dates,
            )

    def dates(self) -> pd.DatetimeIndex:
        all_dates: set = set()
        for df in self.data.values():
            all_dates.update(df.index.tolist())
        return pd.DatetimeIndex(sorted(all_dates))

    def get(self, symbol: str, as_of: pd.Timestamp) -> pd.DataFrame:
        df = self.data[symbol]
        return df[df.index <= as_of]
