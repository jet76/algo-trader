import pandas as pd

from backtester.orders import Signal, SignalType
from backtester.strategy import Strategy
from yfinance_sandbox import bollinger_bands, relative_strength_index


class BollingerRsiStrategy(Strategy):
    """
    Combined mean reversion: buy at lower Bollinger band AND RSI oversold.
    Sell at upper band OR RSI overbought (whichever fires first).
    Higher conviction than either signal alone.
    """

    def __init__(self, bb_window: int = 20, rsi_oversold: int = 40, rsi_overbought: int = 60):
        self.bb_window = bb_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def on_bar(self, symbol: str, data: pd.DataFrame) -> Signal:
        bb = bollinger_bands(data, self.bb_window)
        rsi = relative_strength_index(data)
        price = data["Close"]

        at_lower = price.iloc[-1] <= bb["Lower_Band"].iloc[-1]
        at_upper = price.iloc[-1] >= bb["Upper_Band"].iloc[-1]
        oversold = rsi.iloc[-1] < self.rsi_oversold
        overbought = rsi.iloc[-1] > self.rsi_overbought

        if at_lower and oversold:
            return Signal(symbol=symbol, signal_type=SignalType.BUY)
        if at_upper or overbought:
            return Signal(symbol=symbol, signal_type=SignalType.SELL)
        return Signal(symbol=symbol, signal_type=SignalType.HOLD)
