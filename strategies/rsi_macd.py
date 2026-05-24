import pandas as pd

from backtester.orders import Signal, SignalType
from backtester.strategy import Strategy
from yfinance_sandbox import moving_average_convergence_divergence, relative_strength_index


class RsiMacdStrategy(Strategy):
    def __init__(self, rsi_oversold: int = 30, rsi_overbought: int = 70):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def on_bar(self, symbol: str, data: pd.DataFrame) -> Signal:
        rsi = relative_strength_index(data)
        macd_df = moving_average_convergence_divergence(data)

        current_rsi = rsi.iloc[-1]
        macd = macd_df["MACD"]
        sig = macd_df["Signal_Line"]

        macd_crossed_up = macd.iloc[-2] < sig.iloc[-2] and macd.iloc[-1] > sig.iloc[-1]
        macd_crossed_down = macd.iloc[-2] > sig.iloc[-2] and macd.iloc[-1] < sig.iloc[-1]

        if current_rsi < self.rsi_oversold and macd_crossed_up:
            return Signal(symbol=symbol, signal_type=SignalType.BUY)
        if current_rsi > self.rsi_overbought and macd_crossed_down:
            return Signal(symbol=symbol, signal_type=SignalType.SELL)

        return Signal(symbol=symbol, signal_type=SignalType.HOLD)
