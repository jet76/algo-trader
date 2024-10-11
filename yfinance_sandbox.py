from pprint import pprint
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

"""
Calculating Methods
"""


def accumulation_distribution_line(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the Accumulation/Distribution Line (ADL).
    Formula: ADL = Previous ADL + ((Close - Low) - (High - Close)) / (High - Low) * Volume
    Purpose: Combines price and volume to determine whether a stock is being accumulated (bought) or distributed (sold).
    Example: Rising ADL indicates accumulation (buying pressure).

    :param data: DataFrame with stock data containing 'High', 'Low', 'Close', and 'Volume' columns.
    :return: Series of ADL values.
    """
    money_flow_multiplier = (
        (data["Close"] - data["Low"]) - (data["High"] - data["Close"])
    ) / (data["High"] - data["Low"])
    money_flow_volume = money_flow_multiplier * data["Volume"]
    adl = money_flow_volume.cumsum()
    return adl


def average_true_range(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate the Average True Range (ATR).
    Formula: ATR = Average of True Ranges (True Range = max(high - low, abs(high - previous close), abs(low - previous close)))
    Purpose: Measures market volatility by taking the difference between high and low prices.
    Example: Used to set stop-loss levels or identify volatility trends.

    :param data: DataFrame with stock data containing 'High', 'Low', and 'Close' columns.
    :param window: The period for calculating ATR (default is 14).
    :return: Series of ATR values.
    """
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr


def bollinger_bands(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate the Bollinger Bands.
    Formula: Upper Band = SMA + (2 * Standard Deviation), Lower Band = SMA - (2 * Standard Deviation)
    Purpose: A volatility indicator that defines overbought or oversold levels relative to price. When price touches or crosses these bands, it can signal reversals.
    Example: 20-day SMA with 2 standard deviations is commonly used.

    :param data: DataFrame with stock data containing the 'Close' column.
    :param window: The period for the Bollinger Bands (default is 20).
    :return: DataFrame with Upper Band and Lower Band.
    """
    sma = data["Close"].rolling(window=window).mean()
    std = data["Close"].rolling(window=window).std()
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    return pd.DataFrame({"Upper_Band": upper_band, "Lower_Band": lower_band})


def exponential_moving_average(data: pd.DataFrame, span: int = 50) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA).
    Formula: EMA = (Close - EMA_prev) * (2 / (n + 1)) + EMA_prev
    Purpose: Similar to SMA, but it gives more weight to recent prices, making it more responsive to recent price changes.
    Example: A 50-day EMA is commonly used for longer-term trends.

    :param data: DataFrame with stock data containing the 'Close' column.
    :param span: The period for the EMA.
    :return: Series of EMA values.
    """
    return data["Close"].ewm(span=span, adjust=False).mean()


def moving_average_convergence_divergence(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Moving Average Convergence Divergence (MACD).
    Formula: MACD = 12-day EMA - 26-day EMA; Signal Line = 9-day EMA of MACD
    Purpose: Measures momentum by comparing short-term and long-term EMAs to identify potential buy or sell signals.
    Example: When MACD crosses above the signal line, it can indicate a buying opportunity.

    :param data: DataFrame with stock data containing the 'Close' column.
    :return: DataFrame with MACD and Signal Line.
    """
    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return pd.DataFrame({"MACD": macd, "Signal_Line": signal_line})


def on_balance_volume(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the On-Balance Volume (OBV).
    Formula: OBV = Previous OBV + Volume (if close > previous close) or - Volume (if close < previous close)
    Purpose: Measures buying and selling pressure by using volume as a factor. Helps in identifying divergence between price and volume.
    Example: A rising OBV indicates buying pressure.

    :param data: DataFrame with stock data containing 'Close' and 'Volume' columns.
    :return: Series of OBV values.
    """
    obv = (
        data["Volume"]
        * (
            (data["Close"] > data["Close"].shift(1)).astype(int)
            - (data["Close"] < data["Close"].shift(1)).astype(int)
        )
    ).cumsum()
    return obv


def price_rate_of_change(data: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Calculate the Price Rate of Change (ROC).
    Formula: ROC = [(Close - Close n days ago) / Close n days ago] * 100
    Purpose: Measures the percentage change in price over a specified period, often used to identify overbought or oversold conditions.
    Example: A 10-day ROC is common to detect momentum shifts.

    :param data: DataFrame with stock data containing the 'Close' column.
    :param period: The period for calculating ROC (default is 10).
    :return: Series of ROC values.
    """
    roc = (
        (data["Close"] - data["Close"].shift(period)) / data["Close"].shift(period)
    ) * 100
    return roc


def relative_strength_index(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a given DataFrame of stock prices.
    Formula: RSI = 100 - [100 / (1 + (Average Gain / Average Loss))]
    Purpose: Measures the speed and change of price movements to identify overbought or oversold conditions (values range from 0 to 100, with 70 being overbought and 30 being oversold).
    Example: A 14-day RSI is commonly used.

    :param data: DataFrame containing stock price data with a 'Close' column.
    :type data: pd.DataFrame
    :param window: The number of periods to use for calculating the RSI, default is 14.
    :type window: int
    :return: Series containing the RSI values.
    :rtype: pd.Series
    """
    # Calculate price changes
    delta = data["Close"].diff()

    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    # Calculate the Relative Strength (RS)
    rs = gain / loss

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def simple_moving_average(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate the Simple Moving Average (SMA).
    Formula: SMA = (Sum of the closing prices over a specific period) / Period
    Purpose: Tracks the average price of a stock over a specified number of days to smooth out short-term fluctuations and identify trends.
    Example: A 20-day moving average helps identify short-term trends.

    :param data: DataFrame with stock data containing the 'Close' column.
    :param window: The period for the moving average.
    :return: Series of SMA values.
    """
    return data["Close"].rolling(window=window).mean()


def stochastic_oscillator(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate the Stochastic Oscillator.
    Formula: %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    Purpose: Compares the closing price to a range of prices over a given period. It signals overbought or oversold conditions.
    Example: %K line crossing above the %D line (3-day SMA of %K) can indicate a buy signal.

    :param data: DataFrame with stock data containing 'High', 'Low', and 'Close' columns.
    :param window: The period for calculating the oscillator (default is 14).
    :return: DataFrame with %K and %D values.
    """
    low_14 = data["Low"].rolling(window=window).min()
    high_14 = data["High"].rolling(window=window).max()
    percent_k = (data["Close"] - low_14) * 100 / (high_14 - low_14)
    percent_d = percent_k.rolling(window=3).mean()
    return pd.DataFrame({"%K": percent_k, "%D": percent_d})


def volume_weighted_average_price(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the Volume Weighted Average Price (VWAP).
    Formula: VWAP = (Sum of (Price * Volume)) / Total Volume
    Purpose: Used to assess the average price of a stock weighted by volume. It provides insight into the price levels where the majority of trading has occurred.
    Example: VWAP is used as a trading benchmark and support/resistance level.

    :param data: DataFrame with stock data containing 'Close' and 'Volume' columns.
    :return: Series of VWAP values.
    """
    cumulative_price_volume = (data["Close"] * data["Volume"]).cumsum()
    cumulative_volume = data["Volume"].cumsum()
    vwap = cumulative_price_volume / cumulative_volume
    return vwap


"""
Graphing Methods
"""


def plot_adl(adl: pd.Series) -> None:
    """
    Plot Accumulation/Distribution Line (ADL).

    :param adl: Series representing the Accumulation/Distribution Line.
    :return: None
    """
    plt.figure(figsize=(10, 4))
    plt.plot(adl, label="ADL", color="green")
    plt.title("Accumulation/Distribution Line (ADL)")
    plt.xlabel("Date")
    plt.ylabel("ADL")
    plt.legend()
    plt.show()


def plot_atr(atr: pd.Series) -> None:
    """
    Plot Average True Range (ATR).

    :param atr: Series representing the calculated ATR values.
    :return: None
    """
    plt.figure(figsize=(10, 4))
    plt.plot(atr, label="ATR", color="orange")
    plt.title("Average True Range (ATR)")
    plt.xlabel("Date")
    plt.ylabel("ATR")
    plt.legend()
    plt.show()


def plot_bollinger_bands(
    price_data: pd.Series, upper_band: pd.Series, lower_band: pd.Series
) -> None:
    """
    Plot Bollinger Bands along with price data.

    :param price_data: Series of stock prices (e.g., closing prices).
    :param upper_band: Series representing the upper Bollinger Band.
    :param lower_band: Series representing the lower Bollinger Band.
    :return: None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(price_data, label="Price", color="blue")
    plt.plot(upper_band, label="Upper Band", color="orange")
    plt.plot(lower_band, label="Lower Band", color="green")
    plt.fill_between(price_data.index, lower_band, upper_band, color="gray", alpha=0.2)
    plt.title("Bollinger Bands")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def plot_macd(macd: pd.Series, signal: pd.Series, histogram: pd.Series) -> None:
    """
    Plot Moving Average Convergence Divergence (MACD) along with signal line and histogram.

    :param macd: Series representing MACD values.
    :param signal: Series representing the signal line values.
    :param histogram: Series representing the MACD histogram.
    :return: None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(macd, label="MACD", color="blue")
    plt.plot(signal, label="Signal Line", color="red")
    plt.bar(histogram.index, histogram, color="gray", label="MACD Histogram")
    plt.title("MACD and Signal Line")
    plt.xlabel("Date")
    plt.ylabel("MACD")
    plt.legend()
    plt.show()


def plot_obv(obv: pd.Series) -> None:
    """
    Plot On-Balance Volume (OBV).

    :param obv: Series representing the On-Balance Volume.
    :return: None
    """
    plt.figure(figsize=(10, 4))
    plt.plot(obv, label="OBV", color="green")
    plt.title("On-Balance Volume (OBV)")
    plt.xlabel("Date")
    plt.ylabel("OBV")
    plt.legend()
    plt.show()


def plot_roc(roc: pd.Series) -> None:
    """
    Plot Price Rate of Change (ROC).

    :param roc: Series representing the Price Rate of Change values.
    :return: None
    """
    plt.figure(figsize=(10, 4))
    plt.plot(roc, label="ROC", color="purple")
    plt.title("Price Rate of Change (ROC)")
    plt.xlabel("Date")
    plt.ylabel("ROC")
    plt.legend()
    plt.show()


def plot_rsi(rsi: pd.Series) -> None:
    """
    Plot Relative Strength Index (RSI).

    :param rsi: Series representing the calculated RSI values.
    :return: None
    """
    plt.figure(figsize=(10, 4))
    plt.plot(rsi, label="RSI", color="purple")
    plt.axhline(y=70, color="red", linestyle="--", label="Overbought")
    plt.axhline(y=30, color="green", linestyle="--", label="Oversold")
    plt.title("Relative Strength Index (RSI)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.show()


def plot_sma_ema(price_data: pd.Series, sma: pd.Series, ema: pd.Series) -> None:
    """
    Plot Simple Moving Average (SMA) and Exponential Moving Average (EMA) along with price data.

    :param price_data: Series of stock prices (e.g., closing prices).
    :param sma: Series representing Simple Moving Average (SMA).
    :param ema: Series representing Exponential Moving Average (EMA).
    :return: None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(price_data, label="Price", color="blue")
    plt.plot(sma, label="SMA", color="green")
    plt.plot(ema, label="EMA", color="orange")
    plt.title("Price with SMA and EMA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def plot_stochastic_oscillator(percent_k: pd.Series, percent_d: pd.Series) -> None:
    """
    Plot Stochastic Oscillator (%K and %D).

    :param percent_k: Series representing the %K values.
    :param percent_d: Series representing the %D values.
    :return: None
    """
    plt.figure(figsize=(10, 4))
    plt.plot(percent_k, label="%K", color="blue")
    plt.plot(percent_d, label="%D", color="orange")
    plt.axhline(y=80, color="red", linestyle="--", label="Overbought")
    plt.axhline(y=20, color="green", linestyle="--", label="Oversold")
    plt.title("Stochastic Oscillator")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def plot_vwap(price_data: pd.Series, vwap: pd.Series) -> None:
    """
    Plot Volume Weighted Average Price (VWAP) along with price data.

    :param price_data: Series of stock prices (e.g., closing prices).
    :param vwap: Series representing the Volume Weighted Average Price.
    :return: None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(price_data, label="Price", color="blue")
    plt.plot(vwap, label="VWAP", color="orange")
    plt.title("Price with VWAP")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # print(is_market_open())

    msft = yf.Ticker("MSFT")
    # print(type(msft))
    hist = msft.history(period="6mo")
    # ninety_days = hist.tail(90)
    # avg = ninety_days.mean()
    # print(f"${avg.Open:,.2f}")
    # print(ninety_days.sum().Open / len(ninety_days))

    info = msft.info
    # print(type(info))
    # pprint(info)

    rsi = relative_strength_index(hist).tail()
    print(rsi)
    plot_rsi(rsi)

    sma = simple_moving_average(hist).tail()
    ema = exponential_moving_average(hist).tail()
    # plot_sma_ema(hist["Close"].tail(), sma, ema)
    # print(stochastic_oscillator(hist).tail())
