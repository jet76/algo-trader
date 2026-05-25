"""
Walk-forward validation — avoids overfitting by separating parameter
optimization from performance evaluation:

  For each rolling window:
    1. Grid-search best params on the TRAIN period (in-sample)
    2. Run with those params on the TEST period (out-of-sample)
    3. Report out-of-sample metrics

If a strategy is genuinely robust, train Sharpe and test Sharpe should
correlate. If test degrades badly vs train, the strategy is overfit.
"""

from typing import Any, Dict, List, Type

import pandas as pd

from .data import DataFeed
from .engine import BacktestEngine
from .metrics import compute_metrics
from .optimizer import grid_search
from .strategy import Strategy


def walk_forward(
    strategy_cls: Type[Strategy],
    param_grid: Dict[str, List[Any]],
    symbol: str,
    start: str,
    end: str,
    train_months: int = 6,
    test_months: int = 3,
    optimize_for: str = "sharpe_ratio",
    mock: bool = False,
) -> pd.DataFrame:
    """
    Rolling walk-forward test. Returns one row per window with both
    in-sample (train) best metric and out-of-sample (test) metrics.
    """
    windows = _rolling_windows(start, end, train_months, test_months)
    if not windows:
        raise ValueError(
            f"Date range {start}→{end} too short for "
            f"{train_months}mo train + {test_months}mo test windows."
        )

    rows = []
    for i, (train_start, train_end, test_end) in enumerate(windows):
        ts = train_start.strftime("%Y-%m-%d")
        te = train_end.strftime("%Y-%m-%d")
        oe = test_end.strftime("%Y-%m-%d")

        # ── Optimise on train period ──────────────────────────────────────
        train_results = grid_search(
            strategy_cls=strategy_cls,
            param_grid=param_grid,
            symbols=[symbol],
            start=ts,
            end=te,
            optimize_for=optimize_for,
            mock=mock,
            max_workers=8,
        )
        if train_results.empty or "error" in train_results.columns:
            continue

        best = train_results.iloc[0]
        # Cast back to original types — DataFrame stores int params as float
        best_params = {k: type(param_grid[k][0])(best[k]) for k in param_grid}
        train_metric = float(best[optimize_for])

        # ── Evaluate on test period ───────────────────────────────────────
        feed = DataFeed(symbols=[symbol], start=te, end=oe, mock=mock)
        engine = BacktestEngine(
            feed=feed,
            strategy=strategy_cls(**best_params),
            initial_cash=10_000.0,
            commission=1.0,
            size_pct=0.10,
            slippage=0.001,
        )
        m = compute_metrics(engine.run())

        rows.append(
            {
                "window": i + 1,
                "train": f"{ts} → {te}",
                "test":  f"{te} → {oe}",
                "best_params": str(best_params),
                f"train_{optimize_for}": round(train_metric, 3),
                "test_return_%":   m["total_return_pct"],
                "test_sharpe":     m["sharpe_ratio"],
                "test_drawdown_%": m["max_drawdown_pct"],
                "test_win_rate_%": m["win_rate_pct"],
                "test_trades":     m["total_trades"],
            }
        )

    return pd.DataFrame(rows)


def _rolling_windows(
    start: str, end: str, train_months: int, test_months: int
) -> list:
    windows = []
    cursor = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while True:
        train_end = cursor + pd.DateOffset(months=train_months)
        test_end  = train_end + pd.DateOffset(months=test_months)
        if test_end > end_ts:
            break
        windows.append((cursor, train_end, test_end))
        cursor += pd.DateOffset(months=test_months)
    return windows
