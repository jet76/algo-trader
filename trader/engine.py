import logging
import time
from typing import Dict, List

import pandas as pd

from backtester.orders import Signal, SignalType
from backtester.strategy import Strategy
from broker.base import Broker
from sandbox import is_market_open

log = logging.getLogger(__name__)


class LiveEngine:
    def __init__(
        self,
        symbols: List[str],
        strategy: Strategy,
        broker: Broker,
        warmup_bars: int = 100,
        interval_seconds: int = 300,
        size_pct: float = 0.10,
        exchange: str = "NYSE",
    ):
        self.symbols = symbols
        self.strategy = strategy
        self.broker = broker
        self.warmup_bars = warmup_bars
        self.interval_seconds = interval_seconds
        self.size_pct = size_pct
        self.exchange = exchange
        self.history: Dict[str, pd.DataFrame] = {}

    def run(self) -> None:
        self._warmup()
        log.info("Live loop started — interval %ds, exchange %s", self.interval_seconds, self.exchange)
        while True:
            if not is_market_open(self.exchange):
                log.debug("Market closed, sleeping 60s")
                time.sleep(60)
                continue
            self._tick()
            time.sleep(self.interval_seconds)

    def _warmup(self) -> None:
        log.info("Warming up: fetching %d bars for %s", self.warmup_bars, self.symbols)
        for symbol in self.symbols:
            self.history[symbol] = self.broker.get_bars(symbol, limit=self.warmup_bars)
            log.info("  %s: %d bars loaded", symbol, len(self.history[symbol]))

    def _tick(self) -> None:
        for symbol in self.symbols:
            latest = self.broker.get_bars(symbol, limit=1)
            if latest.empty:
                continue

            hist = self.history[symbol]
            new_rows = latest[~latest.index.isin(hist.index)]
            if not new_rows.empty:
                self.history[symbol] = pd.concat([hist, new_rows]).tail(500)

            data = self.history[symbol]
            if len(data) < 30:
                continue

            signal = self.strategy.on_bar(symbol, data)
            self._handle_signal(signal)

    def _handle_signal(self, signal: Signal) -> None:
        symbol = signal.symbol
        current_qty = self.broker.get_position(symbol)

        if signal.signal_type == SignalType.BUY and current_qty == 0:
            qty = self._size(signal)
            if qty > 0:
                order_id = self.broker.submit_order(symbol, qty, "buy")
                log.info("BUY %d %s — order %s", qty, symbol, order_id)

        elif signal.signal_type == SignalType.SELL and current_qty > 0:
            order_id = self.broker.submit_order(symbol, current_qty, "sell")
            log.info("SELL %d %s — order %s", current_qty, symbol, order_id)

        else:
            log.debug("%s HOLD (qty=%d)", symbol, current_qty)

    def _size(self, signal: Signal) -> int:
        if signal.quantity > 0:
            return signal.quantity
        latest_close = float(self.history[signal.symbol]["Close"].iloc[-1])
        if latest_close <= 0:
            return 0
        return int(self.broker.get_equity() * self.size_pct / latest_close)
