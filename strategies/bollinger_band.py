import pandas as pd

from backtester.orders import Signal, SignalType
from backtester.strategy import Strategy
from yfinance_sandbox import bollinger_bands


class BollingerBandStrategy(Strategy):
    """Mean reversion: buy when price touches the lower band, sell at the upper band."""

    def __init__(self, window: int = 20):
        self.window = window

    def on_bar(self, symbol: str, data: pd.DataFrame) -> Signal:
        bb = bollinger_bands(data, self.window)
        price = data["Close"]

        touched_lower = price.iloc[-2] >= bb["Lower_Band"].iloc[-2] and price.iloc[-1] <= bb["Lower_Band"].iloc[-1]
        touched_upper = price.iloc[-2] <= bb["Upper_Band"].iloc[-2] and price.iloc[-1] >= bb["Upper_Band"].iloc[-1]

        if touched_lower:
            return Signal(symbol=symbol, signal_type=SignalType.BUY)
        if touched_upper:
            return Signal(symbol=symbol, signal_type=SignalType.SELL)
        return Signal(symbol=symbol, signal_type=SignalType.HOLD)
