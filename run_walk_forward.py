"""
Walk-forward validation — tests whether optimized parameters generalize to
unseen data, or whether they're overfit to the training period.

For each rolling window:
  Train period: grid-search best params
  Test period:  run with those params, measure out-of-sample performance

A robust strategy shows consistent test Sharpe across windows.
A fragile strategy shows high train Sharpe but low/negative test Sharpe.

  python run_walk_forward.py
  MOCK_DATA=1 python run_walk_forward.py
"""

import os

import matplotlib.pyplot as plt

from backtester import walk_forward
from strategies.sma_cross import SmaCrossStrategy
from strategies.ema_cross import EmaCrossStrategy
from strategies.rsi_only import RsiOnlyStrategy

SYMBOLS = ["MSFT", "AAPL", "GOOGL", "SPY"]
START   = "2021-01-01"
END     = "2024-01-01"
MOCK    = os.environ.get("MOCK_DATA", "0") == "1"

TESTS = [
    {
        "label": "SmaCrossStrategy",
        "cls": SmaCrossStrategy,
        "grid": {"fast": [5, 10, 20], "slow": [20, 50, 100]},
    },
    {
        "label": "EmaCrossStrategy",
        "cls": EmaCrossStrategy,
        "grid": {"fast": [5, 9, 12], "slow": [20, 26, 50]},
    },
    {
        "label": "RsiOnlyStrategy",
        "cls": RsiOnlyStrategy,
        "grid": {"oversold": [25, 30, 35], "overbought": [65, 70, 75]},
    },
]

all_wf_results = {}

for test in TESTS:
    label = test["label"]
    print(f"\n{'═' * 64}")
    print(f"Walk-Forward: {label}  ({START} → {END})")
    print(f"  Train: 6mo  Test: 3mo  ({'mock' if MOCK else 'real'} data)")
    print(f"{'═' * 64}")

    symbol_results = {}
    for symbol in SYMBOLS:
        print(f"\n  {symbol}")
        df = walk_forward(
            strategy_cls=test["cls"],
            param_grid=test["grid"],
            symbol=symbol,
            start=START,
            end=END,
            train_months=6,
            test_months=3,
            optimize_for="sharpe_ratio",
            mock=MOCK,
        )
        symbol_results[symbol] = df

        if df.empty:
            print("    No windows generated.")
            continue

        cols = ["window", "train", "test", "best_params",
                "train_sharpe_ratio", "test_sharpe", "test_return_%", "test_trades"]
        cols = [c for c in cols if c in df.columns]
        col_w = 14

        header = f"  {'Win':>4}  {'Train period':>22}  {'Test period':>22}  " \
                 f"{'Best params':<26}  {'TrainSharpe':>12}  {'TestSharpe':>10}  {'TestRet%':>9}  {'Trades':>6}"
        print(header)
        print("  " + "─" * (len(header) - 2))
        for _, row in df.iterrows():
            print(
                f"  {int(row['window']):>4}  {row['train']:>22}  {row['test']:>22}  "
                f"{row['best_params']:<26}  {row.get('train_sharpe_ratio', 0):>12.3f}  "
                f"{row['test_sharpe']:>10.3f}  {row['test_return_%']:>+8.2f}%  "
                f"{int(row['test_trades']):>6}"
            )

        avg_train = df["train_sharpe_ratio"].mean() if "train_sharpe_ratio" in df else float("nan")
        avg_test  = df["test_sharpe"].mean()
        avg_ret   = df["test_return_%"].mean()
        print(f"\n  Avg  train Sharpe: {avg_train:.3f}   test Sharpe: {avg_test:.3f}   "
              f"test Return: {avg_ret:+.2f}%")
        degradation = avg_train - avg_test
        verdict = "ROBUST" if degradation < 0.3 else "FRAGILE (overfit warning)"
        print(f"  Sharpe degradation: {degradation:.3f}  →  {verdict}")

    all_wf_results[label] = symbol_results

# ── Plot: train vs test Sharpe across windows ─────────────────────────────────
n_tests  = len(TESTS)
n_syms   = len(SYMBOLS)
fig, axes = plt.subplots(n_tests, n_syms, figsize=(4 * n_syms, 3 * n_tests), sharey=False)
if n_tests == 1:
    axes = [axes]

for r, test in enumerate(TESTS):
    for c, symbol in enumerate(SYMBOLS):
        ax = axes[r][c]
        df = all_wf_results[test["label"]].get(symbol, None)
        if df is not None and not df.empty and "train_sharpe_ratio" in df.columns:
            wins = df["window"]
            ax.plot(wins, df["train_sharpe_ratio"], "o--", color="steelblue",
                    label="Train", linewidth=1.2)
            ax.plot(wins, df["test_sharpe"], "s-", color="tomato",
                    label="Test", linewidth=1.2)
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
            ax.set_xticks(wins)
        ax.set_title(f"{test['label'].replace('Strategy','')}\n{symbol}", fontsize=8)
        if c == 0:
            ax.set_ylabel("Sharpe", fontsize=7)
        if r == 0 and c == n_syms - 1:
            ax.legend(fontsize=7)

fig.suptitle("Walk-Forward: Train vs Test Sharpe by Window", fontsize=11)
fig.tight_layout()
fig.savefig("walk_forward.png", dpi=130, bbox_inches="tight")
print(f"\nChart saved → walk_forward.png")
