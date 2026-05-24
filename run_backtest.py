import matplotlib.pyplot as plt

from backtester import BacktestEngine, DataFeed, compute_metrics
from strategies.rsi_macd import RsiMacdStrategy
from strategies.sma_cross import SmaCrossStrategy

SYMBOL = "MSFT"
START = "2022-01-01"
END = "2024-01-01"
INITIAL_CASH = 10_000.0

import os

USE_MOCK = os.environ.get("MOCK_DATA", "0") == "1"

print(f"Loading data for {SYMBOL} ({START} → {END}){' [mock]' if USE_MOCK else ''}...")
feed = DataFeed(symbols=[SYMBOL], start=START, end=END, mock=USE_MOCK)

strategy = SmaCrossStrategy(fast=20, slow=50)
engine = BacktestEngine(
    feed=feed,
    strategy=strategy,
    initial_cash=INITIAL_CASH,
    commission=1.0,
    shares_per_trade=10,
)

print("Running backtest...")
portfolio = engine.run()
m = compute_metrics(portfolio)

print(f"\nBacktest Results: {SYMBOL}  {START} → {END}")
print("─" * 50)
print(f"Total Return:      {m['total_return_pct']:+.2f}%")
print(f"Annualized Return: {m['annualized_return_pct']:+.2f}%")
print(f"Sharpe Ratio:      {m['sharpe_ratio']:.2f}")
print(f"Max Drawdown:      {m['max_drawdown_pct']:.2f}%")
print(f"Win Rate:          {m['win_rate_pct']:.1f}%")
print(f"Total Trades:      {m['total_trades']}")
print(f"Avg Win:          ${m['avg_win']:.2f}")
print(f"Avg Loss:         ${m['avg_loss']:.2f}")

eq_df = portfolio.equity_df()
plt.figure(figsize=(12, 5))
plt.plot(eq_df.index, eq_df["equity"], label="Portfolio", color="steelblue")
plt.axhline(y=INITIAL_CASH, color="gray", linestyle="--", label="Initial Cash", alpha=0.7)
plt.title(f"Equity Curve — {SYMBOL} RSI/MACD Strategy")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.tight_layout()
plt.savefig("equity_curve.png", dpi=150)
print("\nEquity curve saved → equity_curve.png")
