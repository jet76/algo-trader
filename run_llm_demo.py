"""
LLM strategy single-bar demo.

Loads recent data for a ticker, runs LLMStrategy on the most recent bar,
and prints the full context sent to Claude alongside its decision.

For backtesting with LLMStrategy, use run_llm_backtest.py — but note that
each bar requires one Claude API call, so keep date ranges short.

  python run_llm_demo.py
  python run_llm_demo.py AAPL
"""

import logging
import os
import sys

from backtester.data import DataFeed
from strategies.llm_strategy import LLMStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

symbol = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
USE_MOCK = os.environ.get("MOCK_DATA", "0") == "1"

print(f"Loading data for {symbol}{' [mock]' if USE_MOCK else ''}...")
feed = DataFeed(symbols=[symbol], start="2023-07-01", end="2024-01-01", mock=USE_MOCK)
data = feed.data[symbol]

strategy = LLMStrategy(lookback=15)

print("\n── Context sent to Claude ──────────────────────────────────")
context = strategy._build_context(symbol, data, position=0)
print(context)
print("────────────────────────────────────────────────────────────\n")

print("Querying Claude...")
signal = strategy.on_bar(symbol, data)
print(f"\nSignal returned: {signal.signal_type.value}")
