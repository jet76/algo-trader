"""
Multi-ticker backtest — runs each symbol independently in parallel, then
prints a side-by-side comparison table and a combined equity curve chart.

  python run_multi_backtest.py
  MOCK_DATA=1 python run_multi_backtest.py
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import pandas as pd

from backtester import BacktestEngine, DataFeed, compute_metrics
from strategies.sma_cross import SmaCrossStrategy

SYMBOLS = ["MSFT", "AAPL", "GOOGL", "AMZN"]
START = "2022-01-01"
END = "2024-01-01"
INITIAL_CASH = 10_000.0
USE_MOCK = os.environ.get("MOCK_DATA", "0") == "1"


def backtest_symbol(symbol: str) -> dict:
    feed = DataFeed(symbols=[symbol], start=START, end=END, mock=USE_MOCK)
    engine = BacktestEngine(
        feed=feed,
        strategy=SmaCrossStrategy(fast=20, slow=50),
        initial_cash=INITIAL_CASH,
        commission=1.0,
        size_pct=0.10,
        slippage=0.001,
    )
    portfolio = engine.run()
    m = compute_metrics(portfolio)
    m["symbol"] = symbol
    m["equity_df"] = portfolio.equity_df()

    bnh_data = feed.data[symbol]
    m["bnh_return_pct"] = round(
        (float(bnh_data["Close"].iloc[-1]) - float(bnh_data["Close"].iloc[0]))
        / float(bnh_data["Close"].iloc[0]) * 100,
        2,
    )
    return m


print(f"Backtesting {SYMBOLS} ({START} → {END}){' [mock]' if USE_MOCK else ''}...")

results = {}
with ThreadPoolExecutor(max_workers=len(SYMBOLS), thread_name_prefix="bt") as pool:
    futures = {pool.submit(backtest_symbol, sym): sym for sym in SYMBOLS}
    for future in as_completed(futures):
        sym = futures[future]
        try:
            results[sym] = future.result()
            print(f"  {sym} done")
        except Exception as exc:
            print(f"  {sym} FAILED: {exc}")

# ── Comparison table ──────────────────────────────────────────────────────────
cols = ["symbol", "total_return_pct", "bnh_return_pct", "annualized_return_pct",
        "sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "total_trades"]
labels = ["Symbol", "Return%", "B&H%", "Ann.Ret%", "Sharpe", "MaxDD%", "WinRate%", "Trades"]

col_w = [8, 9, 8, 10, 8, 8, 10, 8]
header = "  ".join(f"{lbl:>{w}}" for lbl, w in zip(labels, col_w))
sep = "  ".join("─" * w for w in col_w)

print(f"\nResults: SMA Cross (20/50)  {START} → {END}")
print(sep)
print(header)
print(sep)

for sym in SYMBOLS:
    if sym not in results:
        continue
    r = results[sym]
    row_vals = [
        r["symbol"],
        f"{r['total_return_pct']:+.2f}%",
        f"{r['bnh_return_pct']:+.2f}%",
        f"{r['annualized_return_pct']:+.2f}%",
        f"{r['sharpe_ratio']:.2f}",
        f"{r['max_drawdown_pct']:.2f}%",
        f"{r['win_rate_pct']:.1f}%",
        str(r["total_trades"]),
    ]
    print("  ".join(f"{v:>{w}}" for v, w in zip(row_vals, col_w)))

print(sep)

# ── Equity curves ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
colors = ["steelblue", "tomato", "seagreen", "darkorange", "purple", "brown"]

for i, sym in enumerate(SYMBOLS):
    if sym not in results:
        continue
    eq = results[sym]["equity_df"]["equity"]
    ax.plot(eq.index, eq.values, label=sym, color=colors[i % len(colors)], linewidth=1.5)

ax.axhline(y=INITIAL_CASH, color="gray", linestyle=":", alpha=0.5, label="Initial Cash")
ax.set_title(f"Equity Curves — SMA Cross (20/50)  {START} → {END}")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value ($)")
ax.legend()
fig.tight_layout()
fig.savefig("equity_curve_multi.png", dpi=150)
print(f"\nEquity curves saved → equity_curve_multi.png")
