"""
Paper trading entry point.

Dry-run mode (default): fetches real market data via yfinance, logs signals,
never submits real orders. Swap DryRunBroker for AlpacaClient once keys are set.

  python run_trader.py

To use real Alpaca paper trading:
  ALPACA_API_KEY=... ALPACA_SECRET_KEY=... python run_trader.py
"""

import logging
import os

from broker.alpaca import AlpacaClient
from broker.dry_run import DryRunBroker
from strategies.sma_cross import SmaCrossStrategy
from trader.engine import LiveEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SYMBOLS = ["MSFT"]
INITIAL_CASH = 10_000.0

has_keys = bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))

if has_keys:
    broker = AlpacaClient(paper=True)
    logging.getLogger().info("Using Alpaca paper trading account")
else:
    broker = DryRunBroker(initial_cash=INITIAL_CASH)
    logging.getLogger().info("No API keys found — running in dry-run mode (yfinance data, no real orders)")

engine = LiveEngine(
    symbols=SYMBOLS,
    strategy=SmaCrossStrategy(fast=20, slow=50),
    broker=broker,
    warmup_bars=100,
    interval_seconds=300,   # check every 5 minutes during market hours
    size_pct=0.10,
)

engine.run()
