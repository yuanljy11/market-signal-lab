
import numpy as np
import pandas as pd

from config import TRADING_DAYS


def cagr(returns: pd.Series) -> float:
  
    r = returns.dropna()
    wealth = float((1 + r).prod())
    years = len(r) / TRADING_DAYS
    return wealth ** (1 / years) - 1 if years > 0 and wealth > 0 else np.nan


def ann_vol(returns: pd.Series) -> float:
    return float(returns.dropna().std() * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    r = returns.dropna() - rf / TRADING_DAYS
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    r = returns.dropna() - rf / TRADING_DAYS
    downside = r[r < 0].std()
    if downside is np.nan or downside == 0 or np.isnan(downside):
        return np.nan
    return float(r.mean() / downside * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:

    wealth = (1 + returns.dropna()).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1).min())


def calmar(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    return cagr(returns) / mdd if mdd > 0 else np.nan


def summary(returns_dict: dict, rf: float = 0.0) -> pd.DataFrame:
   
    rows = {}
    for name, r in returns_dict.items():
        r = r.dropna()
        rows[name] = {
            "CAGR": cagr(r),
            "Ann.Vol": ann_vol(r),
            "Sharpe": sharpe(r, rf),
            "Sortino": sortino(r, rf),
            "MaxDD": max_drawdown(r),
            "Calmar": calmar(r),
        }
    return pd.DataFrame(rows).T.round(3)
