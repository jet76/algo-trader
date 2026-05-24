"""
Sentiment scoring via Claude API.

Each headline is scored bullish (+1), neutral (0), or bearish (-1) weighted by
confidence. The system prompt is cached so repeated calls within a session only
pay the input token cost once.
"""

import json
import logging
from typing import List

import anthropic

log = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None

_SYSTEM_PROMPT = (
    "You are a financial sentiment analyzer. "
    "Given a stock news headline and ticker symbol, classify the sentiment as it "
    "relates to that stock's near-term price outlook. "
    "Respond with only a JSON object — no explanation, no markdown:\n"
    '{"sentiment": "bullish" | "bearish" | "neutral", "confidence": <0.0-1.0>}'
)

_SENTIMENT_SCORE = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def score_headline(symbol: str, headline: str) -> dict:
    """Score one headline. Returns {'sentiment': str, 'confidence': float}."""
    response = _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=60,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f'Ticker: {symbol}\nHeadline: "{headline}"',
            }
        ],
    )
    try:
        return json.loads(response.content[0].text.strip())
    except (json.JSONDecodeError, IndexError, KeyError):
        log.warning("Failed to parse sentiment response for: %s", headline)
        return {"sentiment": "neutral", "confidence": 0.0}


def score_headlines(symbol: str, headlines: List[str]) -> float:
    """
    Score a list of headlines and return an aggregate sentiment score.
    Returns a float in [-1, +1]: negative = bearish, positive = bullish.
    """
    if not headlines:
        return 0.0

    total, count = 0.0, 0
    for headline in headlines:
        result = score_headline(symbol, headline)
        raw = _SENTIMENT_SCORE.get(result.get("sentiment", "neutral"), 0.0)
        confidence = float(result.get("confidence", 0.5))
        total += raw * confidence
        count += 1
        log.debug(
            "  [%+.2f @ %.2f] %s",
            raw,
            confidence,
            headline[:80],
        )

    return total / count if count else 0.0
