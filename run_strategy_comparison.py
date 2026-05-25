"""
Run every strategy against every ticker and print a pivot comparison table.

  python run_strategy_comparison.py
  MOCK_DATA=1 python run_strategy_comparison.py
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from backtester import BacktestEngine, DataFeed, compute_metrics
from strategies.bollinger_band import BollingerBandStrategy
from strategies.obv_trend import ObvTrendStrategy
from strategies.rsi_macd import RsiMacdStrategy
from strategies.rsi_only import RsiOnlyStrategy
from strategies.sma_cross import SmaCrossStrategy
from strategies.stochastic import StochasticStrategy
from strategies.vwap_cross import VwapCrossStrategy

SYMBOLS = ["MSFT", "AAPL", "GOOGL", "AMZN"]
START = "2022-01-01"
END = "2024-01-01"
INITIAL_CASH = 10_000.0
USE_MOCK = os.environ.get("MOCK_DATA", "0") == "1"

STRATEGIES: dict[str, Callable] = {
    "SmaCross(20,50)":   lambda: SmaCrossStrategy(fast=20, slow=50),
    "SmaCross(10,30)":   lambda: SmaCrossStrategy(fast=10, slow=30),
    "RsiMacd":           lambda: RsiMacdStrategy(),
    "RsiOnly":           lambda: RsiOnlyStrategy(),
    "Bollinger(20)":     lambda: BollingerBandStrategy(window=20),
    "VwapCross":         lambda: VwapCrossStrategy(),
    "Stochastic":        lambda: StochasticStrategy(),
    "ObvTrend(20)":      lambda: ObvTrendStrategy(window=20),
}


def run_one(strategy_name: str, symbol: str) -> dict:
    feed = DataFeed(symbols=[symbol], start=START, end=END, mock=USE_MOCK)
    engine = BacktestEngine(
        feed=feed,
        strategy=STRATEGIES[strategy_name](),
        initial_cash=INITIAL_CASH,
        commission=1.0,
        size_pct=0.10,
        slippage=0.001,
    )
    portfolio = engine.run()
    m = compute_metrics(portfolio)
    bnh_data = feed.data[symbol]
    m["bnh_return_pct"] = round(
        (float(bnh_data["Close"].iloc[-1]) - float(bnh_data["Close"].iloc[0]))
        / float(bnh_data["Close"].iloc[0]) * 100, 2,
    )
    m["equity_df"] = portfolio.equity_df()
    return m


print(f"Running {len(STRATEGIES)} strategies × {len(SYMBOLS)} tickers "
      f"({START} → {END}){' [mock]' if USE_MOCK else ''}...\n")

# Collect all (strategy, symbol) results in parallel
results: dict[tuple, dict] = {}
tasks = [(s, sym) for s in STRATEGIES for sym in SYMBOLS]

with ThreadPoolExecutor(max_workers=8, thread_name_prefix="bt") as pool:
    futures = {pool.submit(run_one, s, sym): (s, sym) for s, sym in tasks}
    for future in as_completed(futures):
        key = futures[future]
        try:
            results[key] = future.result()
            print(f"  ✓ {key[0]:<20} {key[1]}")
        except Exception as exc:
            print(f"  ✗ {key[0]:<20} {key[1]}  ERROR: {exc}")

# ── Pivot table: Return % ─────────────────────────────────────────────────────
sym_w = 12
strat_w = 20
col_w = 10

def pivot_table(metric_key: str, label: str, fmt: str) -> None:
    header = f"{'Strategy':<{strat_w}}" + "".join(f"{s:>{col_w}}" for s in SYMBOLS)
    sep = "─" * (strat_w + col_w * len(SYMBOLS))
    print(f"\n{label}")
    print(sep)
    print(header)
    print(sep)
    for strat in STRATEGIES:
        row = f"{strat:<{strat_w}}"
        for sym in SYMBOLS:
            val = results.get((strat, sym), {}).get(metric_key)
            row += f"{fmt.format(val) if val is not None else 'N/A':>{col_w}}"
        print(row)
    # B&H reference row
    print("─" * (strat_w + col_w * len(SYMBOLS)))
    bnh_row = f"{'Buy & Hold':<{strat_w}}"
    for sym in SYMBOLS:
        val = next(
            (v["bnh_return_pct"] for (s, sy), v in results.items() if sy == sym),
            None,
        )
        bnh_row += f"{fmt.format(val) if val is not None else 'N/A':>{col_w}}"
    print(bnh_row)
    print(sep)

pivot_table("total_return_pct",      "Return %",      "{:+.2f}%")
pivot_table("sharpe_ratio",          "Sharpe Ratio",  "{:.2f}")
pivot_table("max_drawdown_pct",      "Max Drawdown %", "{:.2f}%")
pivot_table("win_rate_pct",          "Win Rate %",    "{:.1f}%")
pivot_table("total_trades",          "Total Trades",  "{}")

# ── Equity curve grid ─────────────────────────────────────────────────────────
n_strats = len(STRATEGIES)
n_syms = len(SYMBOLS)
fig, axes = plt.subplots(n_strats, n_syms, figsize=(4 * n_syms, 3 * n_strats), sharex=False)
fig.suptitle(f"Strategy × Ticker  ({START} → {END})", fontsize=13, y=1.01)

for r, strat in enumerate(STRATEGIES):
    for c, sym in enumerate(SYMBOLS):
        ax = axes[r][c]
        key = (strat, sym)
        if key in results:
            eq = results[key]["equity_df"]["equity"]
            color = "steelblue" if eq.iloc[-1] >= INITIAL_CASH else "tomato"
            ax.plot(eq.index, eq.values, color=color, linewidth=1)
            ax.axhline(INITIAL_CASH, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
            ret = results[key]["total_return_pct"]
            ax.set_title(f"{sym}  {ret:+.1f}%", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(strat, fontsize=7, rotation=0, ha="right", labelpad=60)

fig.tight_layout()
fig.savefig("strategy_comparison.png", dpi=130, bbox_inches="tight")
print("\nEquity curve grid saved → strategy_comparison.png")
