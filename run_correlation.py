"""
Strategy correlation analysis.

Runs all strategies against all tickers, computes average daily returns per
strategy, then plots a correlation heatmap. Highly correlated strategies
cluster together — when building a portfolio of strategies, prefer combining
low-correlation strategies for diversification.

  python run_correlation.py
  MOCK_DATA=1 python run_correlation.py
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtester import BacktestEngine, DataFeed, compute_metrics
from strategies.bollinger_band import BollingerBandStrategy
from strategies.bollinger_rsi import BollingerRsiStrategy
from strategies.ema_cross import EmaCrossStrategy
from strategies.macd_only import MacdOnlyStrategy
from strategies.obv_trend import ObvTrendStrategy
from strategies.roc_momentum import RocMomentumStrategy
from strategies.rsi_macd import RsiMacdStrategy
from strategies.rsi_only import RsiOnlyStrategy
from strategies.sma_cross import SmaCrossStrategy
from strategies.stochastic import StochasticStrategy
from strategies.triple_sma import TripleSmaStrategy
from strategies.vwap_cross import VwapCrossStrategy

SYMBOLS = ["MSFT", "AAPL", "GOOGL", "AMZN", "JPM", "XOM", "SPY", "GLD"]
START   = "2022-01-01"
END     = "2024-01-01"
MOCK    = os.environ.get("MOCK_DATA", "0") == "1"

STRATEGIES = {
    "SmaCross(20,50)":    lambda: SmaCrossStrategy(fast=20, slow=50),
    "SmaCross(5,20)":     lambda: SmaCrossStrategy(fast=5, slow=20),
    "EmaCross(12,26)":    lambda: EmaCrossStrategy(fast=12, slow=26),
    "EmaCross(5,20)":     lambda: EmaCrossStrategy(fast=5, slow=20),
    "TripleSma":          lambda: TripleSmaStrategy(),
    "MacdOnly":           lambda: MacdOnlyStrategy(),
    "RsiMacd":            lambda: RsiMacdStrategy(),
    "RocMomentum(10)":    lambda: RocMomentumStrategy(period=10),
    "RsiOnly(30,70)":     lambda: RsiOnlyStrategy(oversold=30, overbought=70),
    "RsiOnly(40,60)":     lambda: RsiOnlyStrategy(oversold=40, overbought=60),
    "Bollinger(20)":      lambda: BollingerBandStrategy(window=20),
    "BollingerRsi":       lambda: BollingerRsiStrategy(),
    "Stochastic(20,80)":  lambda: StochasticStrategy(oversold=20, overbought=80),
    "VwapCross":          lambda: VwapCrossStrategy(),
    "ObvTrend(20)":       lambda: ObvTrendStrategy(window=20),
}


def run_one(strat_name: str, symbol: str) -> tuple[str, str, pd.Series]:
    feed = DataFeed(symbols=[symbol], start=START, end=END, mock=MOCK)
    engine = BacktestEngine(
        feed=feed, strategy=STRATEGIES[strat_name](),
        initial_cash=10_000.0, commission=1.0, size_pct=0.10, slippage=0.001,
    )
    portfolio = engine.run()
    eq = portfolio.equity_df()["equity"]
    daily_ret = eq.pct_change().dropna()
    return strat_name, symbol, daily_ret


print(f"Running {len(STRATEGIES)} strategies × {len(SYMBOLS)} tickers "
      f"({START} → {END}){' [mock]' if MOCK else ''}...")

raw: dict[tuple, pd.Series] = {}
with ThreadPoolExecutor(max_workers=8, thread_name_prefix="corr") as pool:
    futures = {
        pool.submit(run_one, s, sym): (s, sym)
        for s in STRATEGIES for sym in SYMBOLS
    }
    for future in as_completed(futures):
        key = futures[future]
        try:
            sname, sym, ret = future.result()
            raw[key] = ret
            print(f"  ✓ {sname:<22} {sym}")
        except Exception as exc:
            print(f"  ✗ {key}  {exc}")

# ── Average returns per strategy across all tickers ───────────────────────────
strategy_returns = {}
for strat in STRATEGIES:
    series_list = [raw[(strat, sym)] for sym in SYMBOLS if (strat, sym) in raw]
    if series_list:
        combined = pd.concat(series_list, axis=1).mean(axis=1)
        strategy_returns[strat] = combined

returns_df = pd.DataFrame(strategy_returns).dropna()
corr = returns_df.corr()

# ── Print top correlated and least correlated pairs ───────────────────────────
pairs = []
strats = list(corr.columns)
for i in range(len(strats)):
    for j in range(i + 1, len(strats)):
        pairs.append((corr.iloc[i, j], strats[i], strats[j]))

pairs.sort(key=lambda x: abs(x[0]), reverse=True)

print("\nMost correlated strategy pairs (combine these with caution):")
for val, a, b in pairs[:5]:
    print(f"  {val:+.3f}  {a}  ↔  {b}")

print("\nLeast correlated strategy pairs (good candidates for combining):")
for val, a, b in sorted(pairs, key=lambda x: abs(x[0]))[:5]:
    print(f"  {val:+.3f}  {a}  ↔  {b}")

# ── Heatmap ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 11))
n = len(corr)
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(corr.index, fontsize=8)

for i in range(n):
    for j in range(n):
        val = corr.iloc[i, j]
        color = "black" if abs(val) < 0.7 else "white"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=color)

ax.set_title(f"Strategy Return Correlation  ({START} → {END})\n"
             f"averaged across {len(SYMBOLS)} tickers", fontsize=11)
fig.tight_layout()
fig.savefig("strategy_correlation.png", dpi=140, bbox_inches="tight")
print(f"\nHeatmap saved → strategy_correlation.png")
