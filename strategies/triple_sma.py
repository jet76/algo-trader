import pandas as pd

from backtester.orders import Signal, SignalType
from backtester.strategy import Strategy
from yfinance_sandbox import simple_moving_average


class TripleSmaStrategy(Strategy):
    """
    Buy when fast > medium > slow (all three SMAs aligned bullish).
    Sell when fast < medium < slow (all three aligned bearish).
    More selective than a simple two-SMA cross.
    """

    def __init__(self, fast: int = 10, medium: int = 30, slow: int = 60):
        self.fast = fast
        self.medium = medium
        self.slow = slow

    def on_bar(self, symbol: str, data: pd.DataFrame) -> Signal:
        fast = simple_moving_average(data, self.fast)
        med = simple_moving_average(data, self.medium)
        slow = simple_moving_average(data, self.slow)

        prev_bull = fast.iloc[-2] > med.iloc[-2] > slow.iloc[-2]
        curr_bull = fast.iloc[-1] > med.iloc[-1] > slow.iloc[-1]
        prev_bear = fast.iloc[-2] < med.iloc[-2] < slow.iloc[-2]
        curr_bear = fast.iloc[-1] < med.iloc[-1] < slow.iloc[-1]

        if not prev_bull and curr_bull:
            return Signal(symbol=symbol, signal_type=SignalType.BUY)
        if not prev_bear and curr_bear:
            return Signal(symbol=symbol, signal_type=SignalType.SELL)
        return Signal(symbol=symbol, signal_type=SignalType.HOLD)
