import os

import matplotlib.pyplot as plt
import pandas as pd

from backtester import BacktestEngine, DataFeed, compute_metrics
from strategies.rsi_macd import RsiMacdStrategy
from strategies.sma_cross import SmaCrossStrategy

SYMBOL = "MSFT"
START = "2022-01-01"
END = "2024-01-01"
INITIAL_CASH = 10_000.0
USE_MOCK = os.environ.get("MOCK_DATA", "0") == "1"

print(f"Loading data for {SYMBOL} ({START} → {END}){' [mock]' if USE_MOCK else ''}...")
feed = DataFeed(symbols=[SYMBOL], start=START, end=END, mock=USE_MOCK)

strategy = SmaCrossStrategy(fast=20, slow=50)
engine = BacktestEngine(
    feed=feed,
    strategy=strategy,
    initial_cash=INITIAL_CASH,
    commission=1.0,
    size_pct=0.10,
    slippage=0.001,
)

print("Running backtest...")
portfolio = engine.run()
m = compute_metrics(portfolio)

# Buy-and-hold benchmark
symbol_data = feed.data[SYMBOL]
bnh_shares = INITIAL_CASH / float(symbol_data["Close"].iloc[0])
bnh_final = bnh_shares * float(symbol_data["Close"].iloc[-1])
bnh_return = (bnh_final - INITIAL_CASH) / INITIAL_CASH * 100

print(f"\nBacktest Results: {SYMBOL}  {START} → {END}")
print("─" * 52)
print(f"{'':30s} {'Strategy':>10}  {'B&H':>8}")
print("─" * 52)
print(f"{'Total Return':30s} {m['total_return_pct']:>+9.2f}%  {bnh_return:>+7.2f}%")
print(f"{'Annualized Return':30s} {m['annualized_return_pct']:>+9.2f}%")
print(f"{'Sharpe Ratio':30s} {m['sharpe_ratio']:>10.2f}")
print(f"{'Max Drawdown':30s} {m['max_drawdown_pct']:>10.2f}%")
print(f"{'Win Rate':30s} {m['win_rate_pct']:>10.1f}%")
print(f"{'Total Trades':30s} {m['total_trades']:>10}")
print(f"{'Avg Win':30s} ${m['avg_win']:>9.2f}")
print(f"{'Avg Loss':30s} ${m['avg_loss']:>9.2f}")

trade_log = portfolio.trade_log()
if not trade_log.empty:
    print(f"\nTrade Log:")
    print("─" * 72)
    print(
        f"{'Symbol':<8} {'Entry':>12} {'Exit':>12} {'Entry $':>10} "
        f"{'Exit $':>10} {'Qty':>5} {'PnL':>9} {'Ret%':>7}"
    )
    print("─" * 72)
    for _, t in trade_log.iterrows():
        entry = pd.Timestamp(t["entry_date"]).strftime("%Y-%m-%d")
        exit_ = pd.Timestamp(t["exit_date"]).strftime("%Y-%m-%d")
        print(
            f"{t['symbol']:<8} {entry:>12} {exit_:>12} {t['entry_price']:>10.2f} "
            f"{t['exit_price']:>10.2f} {t['qty']:>5} {t['pnl']:>+9.2f} {t['return_pct']:>+6.2f}%"
        )
else:
    print("\nNo completed trades.")

# Equity curve with buy-and-hold comparison
eq_df = portfolio.equity_df()
bnh_curve = INITIAL_CASH * (symbol_data["Close"] / float(symbol_data["Close"].iloc[0]))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(eq_df.index, eq_df["equity"], label="Strategy", color="steelblue", linewidth=1.5)
ax.plot(bnh_curve.index, bnh_curve.values, label="Buy & Hold", color="orange",
        linewidth=1.5, linestyle="--", alpha=0.8)
ax.axhline(y=INITIAL_CASH, color="gray", linestyle=":", alpha=0.5, label="Initial Cash")
ax.set_title(f"Equity Curve — {SYMBOL} ({strategy.__class__.__name__})")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value ($)")
ax.legend()
fig.tight_layout()
fig.savefig("equity_curve.png", dpi=150)
print("\nEquity curve saved → equity_curve.png")
