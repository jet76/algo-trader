import pandas as pd

from backtester.orders import Signal, SignalType
from backtester.strategy import Strategy
from yfinance_sandbox import moving_average_convergence_divergence


class MacdOnlyStrategy(Strategy):
    """Buy when MACD line crosses above signal line; sell on the reverse."""

    def on_bar(self, symbol: str, data: pd.DataFrame) -> Signal:
        df = moving_average_convergence_divergence(data)
        macd = df["MACD"]
        sig = df["Signal_Line"]

        if macd.iloc[-2] < sig.iloc[-2] and macd.iloc[-1] > sig.iloc[-1]:
            return Signal(symbol=symbol, signal_type=SignalType.BUY)
        if macd.iloc[-2] > sig.iloc[-2] and macd.iloc[-1] < sig.iloc[-1]:
            return Signal(symbol=symbol, signal_type=SignalType.SELL)
        return Signal(symbol=symbol, signal_type=SignalType.HOLD)
