import pandas as pd

from backtester.orders import Signal, SignalType
from backtester.strategy import Strategy
from yfinance_sandbox import volume_weighted_average_price


class VwapCrossStrategy(Strategy):
    """Buy when price crosses above cumulative VWAP; sell when it crosses below."""

    def on_bar(self, symbol: str, data: pd.DataFrame) -> Signal:
        vwap = volume_weighted_average_price(data)
        price = data["Close"]

        crossed_above = price.iloc[-2] < vwap.iloc[-2] and price.iloc[-1] > vwap.iloc[-1]
        crossed_below = price.iloc[-2] > vwap.iloc[-2] and price.iloc[-1] < vwap.iloc[-1]

        if crossed_above:
            return Signal(symbol=symbol, signal_type=SignalType.BUY)
        if crossed_below:
            return Signal(symbol=symbol, signal_type=SignalType.SELL)
        return Signal(symbol=symbol, signal_type=SignalType.HOLD)
