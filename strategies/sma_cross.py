import pandas as pd

from backtester.orders import Signal, SignalType
from backtester.strategy import Strategy
from yfinance_sandbox import simple_moving_average


class SmaCrossStrategy(Strategy):
    """Golden cross / death cross: buy when fast SMA crosses above slow SMA, sell on the reverse."""

    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow

    def on_bar(self, symbol: str, data: pd.DataFrame) -> Signal:
        fast_sma = simple_moving_average(data, window=self.fast)
        slow_sma = simple_moving_average(data, window=self.slow)

        if fast_sma.iloc[-2] < slow_sma.iloc[-2] and fast_sma.iloc[-1] > slow_sma.iloc[-1]:
            return Signal(symbol=symbol, signal_type=SignalType.BUY)
        if fast_sma.iloc[-2] > slow_sma.iloc[-2] and fast_sma.iloc[-1] < slow_sma.iloc[-1]:
            return Signal(symbol=symbol, signal_type=SignalType.SELL)

        return Signal(symbol=symbol, signal_type=SignalType.HOLD)
