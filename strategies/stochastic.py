import pandas as pd

from backtester.orders import Signal, SignalType
from backtester.strategy import Strategy
from yfinance_sandbox import stochastic_oscillator


class StochasticStrategy(Strategy):
    """Buy when %K crosses above %D in oversold territory; sell when it crosses below in overbought."""

    def __init__(self, window: int = 14, oversold: int = 20, overbought: int = 80):
        self.window = window
        self.oversold = oversold
        self.overbought = overbought

    def on_bar(self, symbol: str, data: pd.DataFrame) -> Signal:
        stoch = stochastic_oscillator(data, self.window)
        k = stoch["%K"]
        d = stoch["%D"]

        k_crossed_up = k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1]
        k_crossed_down = k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1]

        if k_crossed_up and k.iloc[-1] < self.oversold:
            return Signal(symbol=symbol, signal_type=SignalType.BUY)
        if k_crossed_down and k.iloc[-1] > self.overbought:
            return Signal(symbol=symbol, signal_type=SignalType.SELL)
        return Signal(symbol=symbol, signal_type=SignalType.HOLD)
