from typing import List, Optional

import pandas as pd

from .data import DataFeed
from .orders import Fill, Order, Signal, SignalType
from .portfolio import Portfolio
from .strategy import Strategy


class BacktestEngine:
    def __init__(
        self,
        feed: DataFeed,
        strategy: Strategy,
        initial_cash: float = 10_000.0,
        commission: float = 1.0,
        shares_per_trade: int = 10,
    ):
        self.feed = feed
        self.strategy = strategy
        self.portfolio = Portfolio(initial_cash)
        self.commission = commission
        self.shares_per_trade = shares_per_trade

    def run(self) -> Portfolio:
        dates = self.feed.dates()
        pending_orders: List[Order] = []

        for i, date in enumerate(dates):
            # Execute pending orders at today's open price
            for order in pending_orders:
                symbol_data = self.feed.data[order.symbol]
                if date in symbol_data.index:
                    fill = Fill(
                        symbol=order.symbol,
                        quantity=order.quantity,
                        price=float(symbol_data.loc[date, "Open"]),
                        date=date,
                        commission=self.commission,
                    )
                    self.portfolio.apply_fill(fill)
            pending_orders = []

            # Snapshot current close prices for equity tracking
            current_prices = {
                symbol: float(self.feed.data[symbol].loc[date, "Close"])
                for symbol in self.feed.symbols
                if date in self.feed.data[symbol].index
            }
            self.portfolio.record_equity(date, current_prices)

            # On all bars except the last, run strategy and queue orders for next open
            if i < len(dates) - 1:
                for symbol in self.feed.symbols:
                    data_so_far = self.feed.get(symbol, date)
                    if len(data_so_far) < 30:
                        continue
                    signal = self.strategy.on_bar(symbol, data_so_far)
                    order = self._signal_to_order(signal)
                    if order:
                        pending_orders.append(order)

        return self.portfolio

    def _signal_to_order(self, signal: Signal) -> Optional[Order]:
        current_qty = self.portfolio.current_position(signal.symbol)

        if signal.signal_type == SignalType.BUY and current_qty == 0:
            qty = signal.quantity if signal.quantity > 0 else self.shares_per_trade
            return Order(symbol=signal.symbol, quantity=qty, date=pd.Timestamp.now())
        elif signal.signal_type == SignalType.SELL and current_qty > 0:
            return Order(symbol=signal.symbol, quantity=-current_qty, date=pd.Timestamp.now())

        return None
