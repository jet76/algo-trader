from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

# (daily_drift, daily_volatility) — roughly calibrated to real-world characteristics
# daily_vol = annual_vol / sqrt(252)
_MOCK_PROFILES = {
    # Tech — high vol, strong drift
    "MSFT": (0.00044, 0.016), "AAPL": (0.00036, 0.018), "GOOGL": (0.00036, 0.017),
    "NVDA": (0.00060, 0.030), "META": (0.00036, 0.022),
    # Finance — moderate vol, modest drift
    "JPM":  (0.00028, 0.014), "GS":   (0.00028, 0.016), "BAC":  (0.00020, 0.015),
    # Energy — cyclical, lower drift
    "XOM":  (0.00020, 0.016), "CVX":  (0.00020, 0.015),
    # Healthcare — defensive, low vol
    "JNJ":  (0.00016, 0.010), "PFE":  (0.00012, 0.013), "UNH":  (0.00028, 0.012),
    # Consumer / retail
    "AMZN": (0.00036, 0.022), "TSLA": (0.00024, 0.035), "WMT":  (0.00016, 0.010),
    # ETFs — smoothed by diversification
    "SPY":  (0.00032, 0.010), "QQQ":  (0.00040, 0.013), "DIA":  (0.00028, 0.009),
    "GLD":  (0.00012, 0.007), "TLT":  (0.00004, 0.007), "XLE":  (0.00016, 0.015),
}
_DEFAULT_PROFILE = (0.00028, 0.015)


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
        dates = pd.bdate_range(start=start, end=end)
        for symbol in self.symbols:
            drift, vol = _MOCK_PROFILES.get(symbol, _DEFAULT_PROFILE)
            rng = np.random.default_rng(seed=abs(hash(symbol)) % 2**32)
            n = len(dates)
            close = 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
            high = close * (1 + rng.uniform(0, vol * 1.5, n))
            low  = close * (1 - rng.uniform(0, vol * 1.5, n))
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
