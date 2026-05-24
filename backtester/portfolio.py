from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .orders import Fill


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_cost: float


class Portfolio:
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[dict] = []
        self.fills: List[Fill] = []

    def apply_fill(self, fill: Fill) -> None:
        cost = fill.price * fill.quantity + fill.commission
        self.cash -= cost
        self.fills.append(fill)

        if fill.symbol in self.positions:
            pos = self.positions[fill.symbol]
            new_qty = pos.quantity + fill.quantity
            if new_qty == 0:
                del self.positions[fill.symbol]
            elif fill.quantity > 0:
                total_cost = pos.avg_cost * pos.quantity + fill.price * fill.quantity
                pos.avg_cost = total_cost / new_qty
                pos.quantity = new_qty
            else:
                pos.quantity = new_qty
        elif fill.quantity > 0:
            self.positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=fill.quantity,
                avg_cost=fill.price,
            )

    def record_equity(self, date: pd.Timestamp, prices: Dict[str, float]) -> None:
        market_value = sum(
            pos.quantity * prices.get(pos.symbol, pos.avg_cost)
            for pos in self.positions.values()
        )
        self.equity_curve.append(
            {
                "date": date,
                "equity": self.cash + market_value,
                "cash": self.cash,
                "market_value": market_value,
            }
        )

    def equity_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve).set_index("date")

    def current_position(self, symbol: str) -> int:
        pos = self.positions.get(symbol)
        return pos.quantity if pos else 0

    def trade_log(self) -> pd.DataFrame:
        """Return a DataFrame of completed round-trip trades."""
        rows = []
        open_fills: Dict[str, Fill] = {}
        for fill in self.fills:
            if fill.quantity > 0:
                open_fills[fill.symbol] = fill
            elif fill.quantity < 0 and fill.symbol in open_fills:
                entry = open_fills.pop(fill.symbol)
                qty = abs(fill.quantity)
                pnl = (fill.price - entry.price) * qty - fill.commission - entry.commission
                rows.append(
                    {
                        "symbol": fill.symbol,
                        "entry_date": entry.date,
                        "entry_price": round(entry.price, 4),
                        "exit_date": fill.date,
                        "exit_price": round(fill.price, 4),
                        "qty": qty,
                        "pnl": round(pnl, 2),
                        "return_pct": round((fill.price - entry.price) / entry.price * 100, 2),
                    }
                )
        return pd.DataFrame(rows)
