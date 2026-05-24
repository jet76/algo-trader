from typing import List

import yfinance as yf


def fetch_headlines(symbol: str, limit: int = 10) -> List[str]:
    """Fetch recent news headlines for a symbol via yfinance."""
    news = yf.Ticker(symbol).news
    headlines = []
    for item in (news or [])[:limit]:
        # yfinance >=0.2.40 nests title under 'content'; older versions put it top-level
        title = item.get("content", {}).get("title") or item.get("title", "")
        if title:
            headlines.append(title)
    return headlines
