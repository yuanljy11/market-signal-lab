
import numpy as np
import pandas as pd


def ma_crossover(prices: pd.Series, fast: int = 20, slow: int = 100) -> pd.Series:
    ma_fast = prices.rolling(fast).mean()
    ma_slow = prices.rolling(slow).mean()
    return (ma_fast > ma_slow).astype(float)


def tsmom(prices: pd.Series, lookback: int = 252) -> pd.Series:

    mom = prices / prices.shift(lookback) - 1
    return (mom > 0).astype(float)


def vol_target(position: pd.Series, returns: pd.Series,
               target_vol: float = 0.15, window: int = 20,
               cap: float = 1.0) -> pd.Series:

    realized = returns.rolling(window).std() * np.sqrt(252)
    scale = (target_vol / realized).clip(upper=cap)
    return (position * scale).fillna(0.0)


def xs_momentum(prices: pd.DataFrame, lookback: int = 126,
                top_n: int = 2) -> pd.DataFrame:

    mom = prices / prices.shift(lookback) - 1
    ranks = mom.rank(axis=1, ascending=False)          
    daily_w = (ranks <= top_n).astype(float)
    row_sum = daily_w.sum(axis=1)
    daily_w = daily_w.div(row_sum.where(row_sum > 0), axis=0).fillna(0.0)

    month = pd.Series(prices.index.month, index=prices.index)
    is_month_end = month != month.shift(-1)            
    weights = daily_w.where(is_month_end).ffill().fillna(0.0)
    return weights
