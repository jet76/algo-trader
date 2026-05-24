from abc import ABC, abstractmethod

import pandas as pd

from .orders import Signal


class Strategy(ABC):
    @abstractmethod
    def on_bar(self, symbol: str, data: pd.DataFrame) -> Signal:
        """
        Called on each bar with all history up to and including the current bar.
        Must return a Signal — never receives future data.
        """
        ...
