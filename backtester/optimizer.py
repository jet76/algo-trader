"""
Grid search optimizer — runs all combinations of strategy parameters in parallel
and returns results ranked by a chosen metric (default: Sharpe ratio).
"""

from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Type

import pandas as pd

from .data import DataFeed
from .engine import BacktestEngine
from .metrics import compute_metrics
from .strategy import Strategy


def grid_search(
    strategy_cls: Type[Strategy],
    param_grid: Dict[str, List[Any]],
    symbols: List[str],
    start: str,
    end: str,
    optimize_for: str = "sharpe_ratio",
    initial_cash: float = 10_000.0,
    commission: float = 1.0,
    size_pct: float = 0.10,
    slippage: float = 0.001,
    mock: bool = False,
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Try every combination of params in param_grid.
    Metrics are averaged across all symbols.
    Returns a DataFrame sorted by optimize_for (descending).
    """
    keys = list(param_grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*param_grid.values())]

    def run_combo(params: dict) -> dict:
        metric_rows = []
        for symbol in symbols:
            feed = DataFeed(symbols=[symbol], start=start, end=end, mock=mock)
            engine = BacktestEngine(
                feed=feed,
                strategy=strategy_cls(**params),
                initial_cash=initial_cash,
                commission=commission,
                size_pct=size_pct,
                slippage=slippage,
            )
            metric_rows.append(compute_metrics(engine.run()))

        scalar_keys = [k for k in metric_rows[0] if isinstance(metric_rows[0][k], (int, float))]
        avg = {k: round(sum(r[k] for r in metric_rows) / len(metric_rows), 4) for k in scalar_keys}
        avg.update(params)
        return avg

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="opt") as pool:
        futures = {pool.submit(run_combo, p): p for p in combos}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:
                params = futures[future]
                rows.append({**params, "error": str(exc)})

    df = pd.DataFrame(rows)
    if optimize_for in df.columns:
        df = df.sort_values(optimize_for, ascending=False).reset_index(drop=True)
    return df
