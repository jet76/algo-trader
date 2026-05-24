from typing import Dict

import numpy as np

from .portfolio import Portfolio


def compute_metrics(portfolio: Portfolio) -> Dict:
    eq = portfolio.equity_df()["equity"]
    initial_cash = portfolio.initial_cash

    total_return = (eq.iloc[-1] - initial_cash) / initial_cash * 100

    n_years = len(eq) / 252
    annualized_return = (
        ((eq.iloc[-1] / initial_cash) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0.0
    )

    daily_returns = eq.pct_change().dropna()
    sharpe = (
        float(daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if daily_returns.std() > 0
        else 0.0
    )

    rolling_max = eq.cummax()
    max_drawdown = float(((eq - rolling_max) / rolling_max).min() * 100)

    # Compute trade P&L from fills (long trades only)
    trade_pnl = []
    open_price: Dict[str, float] = {}
    for fill in portfolio.fills:
        if fill.quantity > 0:
            open_price[fill.symbol] = fill.price
        elif fill.quantity < 0 and fill.symbol in open_price:
            pnl = (fill.price - open_price[fill.symbol]) * abs(fill.quantity)
            trade_pnl.append(pnl)
            del open_price[fill.symbol]

    wins = [p for p in trade_pnl if p > 0]
    losses = [p for p in trade_pnl if p < 0]

    return {
        "total_return_pct": round(total_return, 2),
        "annualized_return_pct": round(annualized_return, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "win_rate_pct": round(len(wins) / len(trade_pnl) * 100, 1) if trade_pnl else 0.0,
        "total_trades": len(trade_pnl),
        "avg_win": round(float(np.mean(wins)), 2) if wins else 0.0,
        "avg_loss": round(float(np.mean(losses)), 2) if losses else 0.0,
    }
