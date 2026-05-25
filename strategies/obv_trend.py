import pandas as pd

from backtester.orders import Signal, SignalType
from backtester.strategy import Strategy
from yfinance_sandbox import on_balance_volume


class ObvTrendStrategy(Strategy):
    """Buy when OBV crosses above its SMA (accumulation); sell when it crosses below (distribution)."""

    def __init__(self, window: int = 20):
        self.window = window

    def on_bar(self, symbol: str, data: pd.DataFrame) -> Signal:
        obv = on_balance_volume(data)
        obv_sma = obv.rolling(window=self.window).mean()

        crossed_above = obv.iloc[-2] < obv_sma.iloc[-2] and obv.iloc[-1] > obv_sma.iloc[-1]
        crossed_below = obv.iloc[-2] > obv_sma.iloc[-2] and obv.iloc[-1] < obv_sma.iloc[-1]

        if crossed_above:
            return Signal(symbol=symbol, signal_type=SignalType.BUY)
        if crossed_below:
            return Signal(symbol=symbol, signal_type=SignalType.SELL)
        return Signal(symbol=symbol, signal_type=SignalType.HOLD)
