"""
Grid search parameter optimization.

Runs every combination of parameters for each strategy against a set of
tickers, averages metrics across tickers, and ranks by Sharpe ratio.

  python run_optimization.py
  MOCK_DATA=1 python run_optimization.py
"""

import os
from typing import Any, Dict, List, Type

from backtester import grid_search
from backtester.strategy import Strategy
from strategies.bollinger_band import BollingerBandStrategy
from strategies.ema_cross import EmaCrossStrategy
from strategies.rsi_only import RsiOnlyStrategy
from strategies.sma_cross import SmaCrossStrategy
from strategies.stochastic import StochasticStrategy

SYMBOLS = ["MSFT", "AAPL", "GOOGL", "AMZN", "JPM", "XOM", "SPY", "GLD"]
START   = "2022-01-01"
END     = "2024-01-01"
MOCK    = os.environ.get("MOCK_DATA", "0") == "1"
TOP_N   = 5

SEARCHES: List[Dict] = [
    {
        "label": "SmaCrossStrategy",
        "cls": SmaCrossStrategy,
        "grid": {"fast": [5, 10, 20], "slow": [20, 50, 100, 200]},
    },
    {
        "label": "EmaCrossStrategy",
        "cls": EmaCrossStrategy,
        "grid": {"fast": [5, 9, 12], "slow": [20, 26, 50]},
    },
    {
        "label": "RsiOnlyStrategy",
        "cls": RsiOnlyStrategy,
        "grid": {"oversold": [25, 30, 35, 40], "overbought": [60, 65, 70, 75]},
    },
    {
        "label": "BollingerBandStrategy",
        "cls": BollingerBandStrategy,
        "grid": {"window": [10, 15, 20, 30]},
    },
    {
        "label": "StochasticStrategy",
        "cls": StochasticStrategy,
        "grid": {"window": [10, 14, 21], "oversold": [15, 20, 25], "overbought": [75, 80, 85]},
    },
]


def fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


for search in SEARCHES:
    label = search["label"]
    cls: Type[Strategy] = search["cls"]
    grid: Dict[str, List[Any]] = search["grid"]
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)

    print(f"\n{'─' * 60}")
    print(f"{label}  —  {n_combos} combinations × {len(SYMBOLS)} tickers")
    print(f"{'─' * 60}")

    results = grid_search(
        strategy_cls=cls,
        param_grid=grid,
        symbols=SYMBOLS,
        start=START,
        end=END,
        optimize_for="sharpe_ratio",
        mock=MOCK,
    )

    param_cols = list(grid.keys())
    metric_cols = ["sharpe_ratio", "total_return_pct", "max_drawdown_pct", "win_rate_pct", "total_trades"]
    display_cols = param_cols + metric_cols
    display_cols = [c for c in display_cols if c in results.columns]

    col_w = 16
    header = "".join(f"{c:>{col_w}}" for c in display_cols)
    print(f"{'Rank':>6}  {header}")
    print("─" * (8 + col_w * len(display_cols)))

    for rank, (_, row) in enumerate(results.head(TOP_N).iterrows(), 1):
        line = "".join(f"{fmt(row[c]):>{col_w}}" for c in display_cols)
        print(f"#{rank:>5}  {line}")

    if len(results) > TOP_N:
        print(f"  ... {len(results) - TOP_N} more combinations not shown")
