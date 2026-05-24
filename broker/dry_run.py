import logging
from typing import Dict

import pandas as pd
import yfinance as yf

from .base import Broker

log = logging.getLogger(__name__)


class DryRunBroker(Broker):
    """
    Fetches real market data via yfinance but never submits real orders.
    Tracks paper positions and equity in memory.
    Swap for AlpacaClient once API keys are configured.
    """

    def __init__(self, initial_cash: float = 10_000.0):
        self._cash = initial_cash
        self._positions: Dict[str, int] = {}
        self._avg_cost: Dict[str, float] = {}

    def get_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 100) -> pd.DataFrame:
        df = yf.Ticker(symbol).history(period=f"{limit + 10}d")
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df[["Open", "High", "Low", "Close", "Volume"]].tail(limit)

    def get_position(self, symbol: str) -> int:
        return self._positions.get(symbol, 0)

    def get_equity(self) -> float:
        return self._cash + sum(
            qty * self._avg_cost.get(sym, 0)
            for sym, qty in self._positions.items()
        )

    def submit_order(self, symbol: str, qty: int, side: str) -> str:
        log.info("[DRY RUN] %s %d shares of %s", side.upper(), abs(qty), symbol)
        if side == "buy":
            self._positions[symbol] = self._positions.get(symbol, 0) + qty
        elif side == "sell":
            self._positions[symbol] = max(0, self._positions.get(symbol, 0) - qty)
        return f"dry-{side}-{symbol}-{qty}"
