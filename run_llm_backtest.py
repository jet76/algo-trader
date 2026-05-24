"""
Backtest with LLMStrategy (Claude makes every BUY/SELL/HOLD decision).

Each bar = one Claude API call. Cost estimate is printed before running.
Default window is 3 months to keep costs reasonable during exploration.

  python run_llm_backtest.py
  MOCK_DATA=1 python run_llm_backtest.py   # no API calls, mock prices
"""

import logging
import os

import matplotlib.pyplot as plt

from backtester import BacktestEngine, DataFeed, compute_metrics
from strategies.llm_strategy import LLMStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")

SYMBOL = "MSFT"
START = "2023-10-01"
END = "2024-01-01"
INITIAL_CASH = 10_000.0
USE_MOCK = os.environ.get("MOCK_DATA", "0") == "1"

print(f"Loading data for {SYMBOL} ({START} → {END}){' [mock]' if USE_MOCK else ''}...")
feed = DataFeed(symbols=[SYMBOL], start=START, end=END, mock=USE_MOCK)
n_bars = len(feed.data[SYMBOL])

if not USE_MOCK:
    est_cost = n_bars * 0.0008   # ~$0.0008 per call at Haiku rates with caching
    print(f"\nEstimated API cost: ~${est_cost:.2f}  ({n_bars} bars × Haiku + prompt caching)")
    confirm = input("Continue? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        raise SystemExit(0)

portfolio_ref = {"obj": None}

strategy = LLMStrategy(
    lookback=15,
    position_fn=lambda symbol: (
        portfolio_ref["obj"].current_position(symbol) if portfolio_ref["obj"] else 0
    ),
)

engine = BacktestEngine(
    feed=feed,
    strategy=strategy,
    initial_cash=INITIAL_CASH,
    commission=1.0,
    size_pct=0.10,
    slippage=0.001,
)

# Wire position_fn to the live portfolio once engine is constructed
portfolio_ref["obj"] = engine.portfolio

print("Running backtest...")
portfolio = engine.run()
m = compute_metrics(portfolio)

bnh_data = feed.data[SYMBOL]
bnh_return = (float(bnh_data["Close"].iloc[-1]) - float(bnh_data["Close"].iloc[0])) / float(bnh_data["Close"].iloc[0]) * 100

print(f"\nBacktest Results: {SYMBOL}  {START} → {END}  (LLMStrategy)")
print("─" * 52)
print(f"{'':30s} {'Strategy':>10}  {'B&H':>8}")
print("─" * 52)
print(f"{'Total Return':30s} {m['total_return_pct']:>+9.2f}%  {bnh_return:>+7.2f}%")
print(f"{'Annualized Return':30s} {m['annualized_return_pct']:>+9.2f}%")
print(f"{'Sharpe Ratio':30s} {m['sharpe_ratio']:>10.2f}")
print(f"{'Max Drawdown':30s} {m['max_drawdown_pct']:>10.2f}%")
print(f"{'Win Rate':30s} {m['win_rate_pct']:>10.1f}%")
print(f"{'Total Trades':30s} {m['total_trades']:>10}")

trade_log = portfolio.trade_log()
if not trade_log.empty:
    import pandas as pd
    print(f"\nTrade Log:")
    print("─" * 72)
    print(f"{'Entry':>12} {'Exit':>12} {'Entry $':>10} {'Exit $':>10} {'Qty':>5} {'PnL':>9} {'Ret%':>7}")
    print("─" * 72)
    for _, t in trade_log.iterrows():
        entry = pd.Timestamp(t["entry_date"]).strftime("%Y-%m-%d")
        exit_ = pd.Timestamp(t["exit_date"]).strftime("%Y-%m-%d")
        print(f"{entry:>12} {exit_:>12} {t['entry_price']:>10.2f} {t['exit_price']:>10.2f} "
              f"{t['qty']:>5} {t['pnl']:>+9.2f} {t['return_pct']:>+6.2f}%")

eq_df = portfolio.equity_df()
bnh_curve = INITIAL_CASH * (bnh_data["Close"] / float(bnh_data["Close"].iloc[0]))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(eq_df.index, eq_df["equity"], label="LLM Strategy", color="steelblue", linewidth=1.5)
ax.plot(bnh_curve.index, bnh_curve.values, label="Buy & Hold", color="orange",
        linewidth=1.5, linestyle="--", alpha=0.8)
ax.axhline(y=INITIAL_CASH, color="gray", linestyle=":", alpha=0.5)
ax.set_title(f"Equity Curve — {SYMBOL} LLM Strategy")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value ($)")
ax.legend()
fig.tight_layout()
fig.savefig("equity_curve_llm.png", dpi=150)
print("\nEquity curve saved → equity_curve_llm.png")
