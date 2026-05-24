"""
Simple TTL cache for sentiment scores. Avoids calling the Claude API on every bar
for the same ticker — scores are only refreshed after `ttl_seconds` have elapsed.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


@dataclass
class _Entry:
    score: float
    expires_at: pd.Timestamp


class SentimentCache:
    def __init__(self, ttl_seconds: int = 4 * 3600):
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, _Entry] = {}

    def get(self, symbol: str) -> Optional[float]:
        entry = self._store.get(symbol)
        if entry and pd.Timestamp.now() < entry.expires_at:
            return entry.score
        return None

    def set(self, symbol: str, score: float) -> None:
        self._store[symbol] = _Entry(
            score=score,
            expires_at=pd.Timestamp.now() + pd.Timedelta(seconds=self.ttl_seconds),
        )
