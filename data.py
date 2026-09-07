
from pathlib import Path

import pandas as pd


def load_prices(tickers, start, cache="data/prices.csv") -> pd.DataFrame:
    p = Path(cache)
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True)

    import yfinance as yf  

    prices = yf.download(tickers, start=start, auto_adjust=True)["Close"]
    prices = prices[tickers]           
    p.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(p)
    return prices


def sanity_check(prices: pd.DataFrame) -> None:

    assert prices.notna().mean().min() > 0.95, 
    daily_move = (prices / prices.shift(1) - 1).abs()

    assert (daily_move.max() < 0.20).all(), f"have suspecious one:\n{daily_move.max()}"
