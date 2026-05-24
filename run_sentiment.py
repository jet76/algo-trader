"""
Standalone sentiment pipeline demo.

Fetches recent headlines for a ticker via yfinance, scores each one with
Claude, and prints the aggregate sentiment.

  python run_sentiment.py
  python run_sentiment.py AAPL 10
"""

import logging
import sys

from sentiment import fetch_headlines, score_headline, score_headlines

logging.basicConfig(level=logging.INFO, format="%(message)s")

symbol = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 8

print(f"Fetching headlines for {symbol}...")
headlines = fetch_headlines(symbol, limit=limit)

if not headlines:
    print("No headlines found.")
    sys.exit(0)

print(f"\n{len(headlines)} headlines found:\n")

results = []
for headline in headlines:
    result = score_headline(symbol, headline)
    sentiment = result.get("sentiment", "neutral")
    confidence = float(result.get("confidence", 0.0))
    marker = {"bullish": "+", "bearish": "-", "neutral": "~"}[sentiment]
    print(f"  [{marker}] ({confidence:.2f}) {headline}")
    results.append(result)

aggregate = score_headlines(symbol, headlines)
label = "BULLISH" if aggregate > 0.1 else "BEARISH" if aggregate < -0.1 else "NEUTRAL"
print(f"\nAggregate sentiment: {aggregate:+.3f}  →  {label}")
