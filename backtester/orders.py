from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    symbol: str
    signal_type: SignalType
    quantity: int = 0  # 0 = use engine default sizing


@dataclass
class Order:
    symbol: str
    quantity: int  # positive = buy, negative = sell
    date: pd.Timestamp


@dataclass
class Fill:
    symbol: str
    quantity: int
    price: float
    date: pd.Timestamp
    commission: float
