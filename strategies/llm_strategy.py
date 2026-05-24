"""
LLM-powered trading strategy using Claude.

Formats a bar's price action and technical indicators into a structured prompt,
asks Claude for a BUY/SELL/HOLD decision, and returns the corresponding Signal.

Position awareness: pass `position_fn` so Claude knows whether a long is open.
Prompt caching: the system prompt is marked ephemeral so it's only billed once
per session rather than on every bar.
"""

import json
import logging
from typing import Callable, Optional

import anthropic
import pandas as pd

from backtester.orders import Signal, SignalType
from backtester.strategy import Strategy
from yfinance_sandbox import (
    bollinger_bands,
    exponential_moving_average,
    moving_average_convergence_divergence,
    relative_strength_index,
    simple_moving_average,
)

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an algorithmic trading assistant analyzing daily stock data.

Given a snapshot of recent price action and technical indicators, decide whether to:
- BUY  — clear bullish setup: positive momentum, not overbought, upward trend
- SELL — clear bearish setup: negative momentum, overbought, trend weakening
- HOLD — mixed or unclear signals; default when uncertain

Risk rule: only recommend SELL when a long position is open. Only recommend BUY
when flat. Prefer HOLD over marginal signals.

Respond with only a JSON object — no markdown, no explanation outside it:
{"action": "buy" | "sell" | "hold", "confidence": 0.0-1.0, "reasoning": "one sentence"}\
"""


class LLMStrategy(Strategy):
    def __init__(
        self,
        lookback: int = 15,
        model: str = "claude-haiku-4-5-20251001",
        position_fn: Optional[Callable[[str], int]] = None,
    ):
        self.lookback = lookback
        self.model = model
        self._position_fn = position_fn or (lambda _: 0)
        self._client: Optional[anthropic.Anthropic] = None

    def on_bar(self, symbol: str, data: pd.DataFrame) -> Signal:
        position = self._position_fn(symbol)
        context = self._build_context(symbol, data, position)
        action, confidence, reasoning = self._query_claude(context)
        log.info("%s → %s (conf=%.2f)  %s", symbol, action.upper(), confidence, reasoning)
        signal_type = {"buy": SignalType.BUY, "sell": SignalType.SELL}.get(action, SignalType.HOLD)
        return Signal(symbol=symbol, signal_type=signal_type)

    # ------------------------------------------------------------------ #

    def _build_context(self, symbol: str, data: pd.DataFrame, position: int) -> str:
        recent = data.tail(self.lookback)
        closes = recent["Close"]
        price = float(closes.iloc[-1])
        today = str(data.index[-1])[:10]

        rsi = float(relative_strength_index(data).iloc[-1])

        macd_df = moving_average_convergence_divergence(data)
        macd_val = float(macd_df["MACD"].iloc[-1])
        sig_val = float(macd_df["Signal_Line"].iloc[-1])

        sma20 = float(simple_moving_average(data, 20).iloc[-1])
        sma50 = float(simple_moving_average(data, 50).iloc[-1])
        ema50 = float(exponential_moving_average(data, 50).iloc[-1])

        bb = bollinger_bands(data, 20)
        bb_upper = float(bb["Upper_Band"].iloc[-1])
        bb_lower = float(bb["Lower_Band"].iloc[-1])
        bb_pct = (
            (price - bb_lower) / (bb_upper - bb_lower) * 100
            if bb_upper != bb_lower else 50.0
        )

        price_chg = (price - float(closes.iloc[0])) / float(closes.iloc[0]) * 100

        lines = [
            f"Symbol: {symbol}",
            f"Date: {today}",
            "",
            f"Recent closes (last {self.lookback} bars):",
        ]
        for date, close in closes.items():
            lines.append(f"  {str(date)[:10]}: ${float(close):.2f}")

        lines += [
            "",
            f"Current price:    ${price:.2f}  ({price_chg:+.1f}% over window)",
            "",
            "Indicators:",
            f"  RSI(14):         {rsi:.1f}",
            f"  MACD:            {macd_val:.3f}  Signal: {sig_val:.3f}  Histogram: {macd_val - sig_val:+.3f}",
            f"  Bollinger %B:    {bb_pct:.1f}  (0=lower band, 100=upper)",
            f"  SMA(20):         ${sma20:.2f}  ({(price - sma20) / sma20 * 100:+.2f}% from price)",
            f"  SMA(50):         ${sma50:.2f}  ({(price - sma50) / sma50 * 100:+.2f}% from price)",
            f"  EMA(50):         ${ema50:.2f}",
            "",
            f"Position: {'LONG (open)' if position > 0 else 'FLAT (no position)'}",
        ]
        return "\n".join(lines)

    def _query_claude(self, context: str) -> tuple[str, float, str]:
        try:
            response = self._get_client().messages.create(
                model=self.model,
                max_tokens=120,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": context}],
            )
            result = json.loads(response.content[0].text.strip())
            return (
                str(result.get("action", "hold")).lower(),
                float(result.get("confidence", 0.5)),
                str(result.get("reasoning", "")),
            )
        except Exception as exc:
            log.warning("Claude query failed (%s) — defaulting to HOLD", exc)
            return "hold", 0.0, ""

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client
