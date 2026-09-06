"""指标层：输入日收益序列，输出各评价指标。

约定：所有函数只接收"日收益 Series"，不接收价格。
这样每个函数都是纯函数，容易单元测试。
"""
import numpy as np
import pandas as pd

from config import TRADING_DAYS


def cagr(returns: pd.Series) -> float:
    """复合年化增长率：终点财富开 1/年数 次方根。
    例：4年翻倍 -> 2**(1/4)-1 = 18.9%/年，而不是 100/4 = 25%。"""
    r = returns.dropna()
    wealth = float((1 + r).prod())
    years = len(r) / TRADING_DAYS
    return wealth ** (1 / years) - 1 if years > 0 and wealth > 0 else np.nan


def ann_vol(returns: pd.Series) -> float:
    """年化波动率 = 日标准差 * sqrt(252)。方差可加 -> 标准差按根号时间放大。"""
    return float(returns.dropna().std() * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """年化Sharpe = 日均超额收益/日波动 * sqrt(252)。"""
    r = returns.dropna() - rf / TRADING_DAYS
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    """Sortino：分母只用下行波动 —— 只惩罚往下的晃动。"""
    r = returns.dropna() - rf / TRADING_DAYS
    downside = r[r < 0].std()
    if downside is np.nan or downside == 0 or np.isnan(downside):
        return np.nan
    return float(r.mean() / downside * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    """最大回撤：沿财富曲线，找"距离历史最高点最深的一摔"。
    wealth: 1.00 -> 1.10 -> 0.99 -> 1.05
    peak:   1.00    1.10    1.10    1.10
    dd:      0%      0%    -10%    -4.5%   -> max_drawdown = -10%
    """
    wealth = (1 + returns.dropna()).cumprod()
    peak = wealth.cummax()
    return float((wealth / peak - 1).min())


def calmar(returns: pd.Series) -> float:
    """Calmar = 年化收益 / |最大回撤|：每承受1单位"最惨一摔"换多少年收益。"""
    mdd = abs(max_drawdown(returns))
    return cagr(returns) / mdd if mdd > 0 else np.nan


def summary(returns_dict: dict, rf: float = 0.0) -> pd.DataFrame:
    """把多个策略的日收益字典压成一张对比表（README 的核心表格）。"""
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
