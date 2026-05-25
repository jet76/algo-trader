import pandas as pd

from backtester.orders import Signal, SignalType
from backtester.strategy import Strategy
from yfinance_sandbox import relative_strength_index


class RsiOnlyStrategy(Strategy):
    """Buy when RSI crosses back up through oversold; sell when it crosses back down through overbought."""

    def __init__(self, oversold: int = 30, overbought: int = 70, window: int = 14):
        self.oversold = oversold
        self.overbought = overbought
        self.window = window

    def on_bar(self, symbol: str, data: pd.DataFrame) -> Signal:
        rsi = relative_strength_index(data, self.window)

        crossed_up = rsi.iloc[-2] <= self.oversold and rsi.iloc[-1] > self.oversold
        crossed_down = rsi.iloc[-2] >= self.overbought and rsi.iloc[-1] < self.overbought

        if crossed_up:
            return Signal(symbol=symbol, signal_type=SignalType.BUY)
        if crossed_down:
            return Signal(symbol=symbol, signal_type=SignalType.SELL)
        return Signal(symbol=symbol, signal_type=SignalType.HOLD)
