import pandas as pd

from backtester.orders import Signal, SignalType
from backtester.strategy import Strategy
from yfinance_sandbox import exponential_moving_average


class EmaCrossStrategy(Strategy):
    """Buy when fast EMA crosses above slow EMA; sell on the reverse."""

    def __init__(self, fast: int = 12, slow: int = 26):
        self.fast = fast
        self.slow = slow

    def on_bar(self, symbol: str, data: pd.DataFrame) -> Signal:
        fast_ema = exponential_moving_average(data, span=self.fast)
        slow_ema = exponential_moving_average(data, span=self.slow)

        if fast_ema.iloc[-2] < slow_ema.iloc[-2] and fast_ema.iloc[-1] > slow_ema.iloc[-1]:
            return Signal(symbol=symbol, signal_type=SignalType.BUY)
        if fast_ema.iloc[-2] > slow_ema.iloc[-2] and fast_ema.iloc[-1] < slow_ema.iloc[-1]:
            return Signal(symbol=symbol, signal_type=SignalType.SELL)
        return Signal(symbol=symbol, signal_type=SignalType.HOLD)
