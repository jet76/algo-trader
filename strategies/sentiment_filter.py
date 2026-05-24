"""
SentimentFilter wraps any existing Strategy and gates BUY signals behind a
minimum sentiment threshold. SELL signals always pass through (don't want
sentiment blocking an exit).

Usage:
    base = SmaCrossStrategy(fast=20, slow=50)
    strategy = SentimentFilter(base, threshold=0.1, headlines=5)
"""

import logging

import pandas as pd

from backtester.orders import Signal, SignalType
from backtester.strategy import Strategy
from sentiment import SentimentCache, fetch_headlines, score_headlines

log = logging.getLogger(__name__)


class SentimentFilter(Strategy):
    def __init__(
        self,
        base_strategy: Strategy,
        threshold: float = 0.0,
        headlines: int = 8,
        ttl_seconds: int = 4 * 3600,
    ):
        self.base = base_strategy
        self.threshold = threshold
        self.headlines = headlines
        self._cache = SentimentCache(ttl_seconds=ttl_seconds)

    def on_bar(self, symbol: str, data: pd.DataFrame) -> Signal:
        signal = self.base.on_bar(symbol, data)

        if signal.signal_type != SignalType.BUY:
            return signal

        score = self._sentiment(symbol)
        log.info("%s sentiment score: %+.2f (threshold %+.2f)", symbol, score, self.threshold)

        if score >= self.threshold:
            return signal

        log.info("%s BUY blocked by sentiment (score %+.2f < %+.2f)", symbol, score, self.threshold)
        return Signal(symbol=symbol, signal_type=SignalType.HOLD)

    def _sentiment(self, symbol: str) -> float:
        cached = self._cache.get(symbol)
        if cached is not None:
            return cached
        headlines = fetch_headlines(symbol, limit=self.headlines)
        score = score_headlines(symbol, headlines)
        self._cache.set(symbol, score)
        return score
