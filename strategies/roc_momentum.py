import pandas as pd

from backtester.orders import Signal, SignalType
from backtester.strategy import Strategy
from yfinance_sandbox import price_rate_of_change


class RocMomentumStrategy(Strategy):
    """Buy when ROC crosses above zero (positive momentum); sell when it crosses below."""

    def __init__(self, period: int = 10):
        self.period = period

    def on_bar(self, symbol: str, data: pd.DataFrame) -> Signal:
        roc = price_rate_of_change(data, period=self.period)

        crossed_positive = roc.iloc[-2] <= 0 and roc.iloc[-1] > 0
        crossed_negative = roc.iloc[-2] >= 0 and roc.iloc[-1] < 0

        if crossed_positive:
            return Signal(symbol=symbol, signal_type=SignalType.BUY)
        if crossed_negative:
            return Signal(symbol=symbol, signal_type=SignalType.SELL)
        return Signal(symbol=symbol, signal_type=SignalType.HOLD)
