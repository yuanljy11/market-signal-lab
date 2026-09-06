"""数据层：下载价格并缓存到本地 CSV。

为什么要缓存：
1. 可复现 —— 别人 clone 你的 repo，结果不随 Yahoo 数据修订而变
2. 不用每次跑分析都打网络请求（yfinance 有时抽风）
这是数据工程的基本素养：raw data 只取一次，落盘，下游全部读缓存。
"""
from pathlib import Path

import pandas as pd


def load_prices(tickers, start, cache="data/prices.csv") -> pd.DataFrame:
    p = Path(cache)
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True)

    import yfinance as yf  # 只有真的要下载时才 import

    prices = yf.download(tickers, start=start, auto_adjust=True)["Close"]
    prices = prices[tickers]            # 固定列顺序
    p.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(p)
    return prices


def sanity_check(prices: pd.DataFrame) -> None:
    """拿到数据先体检，脏数据进回测 = 垃圾进垃圾出。"""
    assert prices.notna().mean().min() > 0.95, "某列缺失值超过5%，检查数据源"
    daily_move = (prices / prices.shift(1) - 1).abs()
    # ETF 单日涨跌超过 20% 几乎必是数据错误（拆股/错价）
    assert (daily_move.max() < 0.20).all(), f"存在可疑跳变:\n{daily_move.max()}"
