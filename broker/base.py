from abc import ABC, abstractmethod

import pandas as pd


class Broker(ABC):
    @abstractmethod
    def get_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 100) -> pd.DataFrame:
        """Return OHLCV DataFrame with DatetimeIndex, most recent `limit` bars."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> int:
        """Return current share count held (0 if none)."""
        ...

    @abstractmethod
    def get_equity(self) -> float:
        """Return total account equity (cash + market value of positions)."""
        ...

    @abstractmethod
    def submit_order(self, symbol: str, qty: int, side: str) -> str:
        """Submit a market order. side='buy' or 'sell'. Returns order ID."""
        ...
