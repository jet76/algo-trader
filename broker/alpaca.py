import os

import pandas as pd
import requests

from .base import Broker

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


class AlpacaClient(Broker):
    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ):
        self._api_key = api_key or os.environ["ALPACA_API_KEY"]
        self._secret_key = secret_key or os.environ["ALPACA_SECRET_KEY"]
        self._base = PAPER_URL if paper else LIVE_URL
        self._headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
        }

    def get_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 100) -> pd.DataFrame:
        resp = requests.get(
            f"{DATA_URL}/v2/stocks/{symbol}/bars",
            headers=self._headers,
            params={"timeframe": timeframe, "limit": limit, "feed": "iex", "adjustment": "raw"},
        )
        resp.raise_for_status()
        bars = resp.json().get("bars", [])
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars).rename(
            columns={"t": "Date", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}
        )
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        return df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]

    def get_position(self, symbol: str) -> int:
        resp = requests.get(f"{self._base}/v2/positions/{symbol}", headers=self._headers)
        if resp.status_code == 404:
            return 0
        resp.raise_for_status()
        return int(resp.json()["qty"])

    def get_equity(self) -> float:
        resp = requests.get(f"{self._base}/v2/account", headers=self._headers)
        resp.raise_for_status()
        return float(resp.json()["equity"])

    def submit_order(self, symbol: str, qty: int, side: str) -> str:
        resp = requests.post(
            f"{self._base}/v2/orders",
            headers=self._headers,
            json={
                "symbol": symbol,
                "qty": str(abs(qty)),
                "side": side,
                "type": "market",
                "time_in_force": "day",
            },
        )
        resp.raise_for_status()
        return resp.json()["id"]
