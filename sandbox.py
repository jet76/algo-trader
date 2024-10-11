from dataclasses import dataclass
from pprint import pprint

import numpy as np
import pandas_market_calendars as mcal
import yfinance as yf


@dataclass
class Security:
    targetHighPrice: float
    targetLowPrice: float
    targetMeanPrice: float
    targetMedianPrice: float
    timeZoneShortName: str


def is_market_open(exchange="NYSE"):
    """
    Determines whether the market is open
    """
    # Get the merket calendar
    calendar = mcal.get_calendar(exchange)
    # Get the current datetime
    now = np.datetime64("today", "s")
    # Get the market schedule for the date
    schedule = calendar.schedule(start_date=now, end_date=now)
    if schedule.empty:
        return False
    else:
        open = schedule.market_open.values[0]
        close = schedule.market_close.values[0]
        return open <= now <= close


if __name__ == "__main__":
    # print(is_market_open())

    msft = yf.Ticker("MSFT")
    print(type(msft))
    # hist = msft.history(period="6mo")
    # ninety_days = hist.tail(90)
    # avg = ninety_days.mean()
    # print(f"${avg.Open:,.2f}")
    # print(ninety_days.sum().Open / len(ninety_days))
    # pprint(msft.info)

    info = msft.info
    # print(type(info))
    # pprint(info)

    # selected = {k: v for k, v in info.items() if k in Security.__dataclass_fields__}
    # dc = Security(**selected)
    # print(dc)
