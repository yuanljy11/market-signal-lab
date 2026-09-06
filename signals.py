"""信号层：每个函数输入价格，输出"仓位"。

约定：
- 单资产信号返回 Series，取值是目标仓位（0=空仓, 1=满仓, 0.5=半仓...）
- 组合信号返回 DataFrame（每列一个资产的权重，行和<=1）
- 信号只允许用"当天及以前"的数据（rolling/pct_change 天然满足）。
  "今天的信号明天才能吃到收益"由 backtest 里的 shift(1) 统一执行，
  信号层不做 shift —— 职责分离，方便单独测试。
"""
import numpy as np
import pandas as pd


def ma_crossover(prices: pd.Series, fast: int = 20, slow: int = 100) -> pd.Series:
    """基线策略：快均线在慢均线上方 -> 持有。"""
    ma_fast = prices.rolling(fast).mean()
    ma_slow = prices.rolling(slow).mean()
    return (ma_fast > ma_slow).astype(float)


def tsmom(prices: pd.Series, lookback: int = 252) -> pd.Series:
    """时序动量 (Moskowitz, Ooi & Pedersen 2012)：
    过去 lookback 天累计涨 -> 持有；累计跌 -> 空仓。
    和MA的区别：只看"起点到今天"的总变化，不平滑中间路径。"""
    mom = prices / prices.shift(lookback) - 1
    return (mom > 0).astype(float)


def vol_target(position: pd.Series, returns: pd.Series,
               target_vol: float = 0.15, window: int = 20,
               cap: float = 1.0) -> pd.Series:
    """波动率目标化：风险控制层，套在任何 0/1 信号外面。

    仓位 = 原信号 * min(目标波动/近期实际波动, cap)
    例：目标15%，最近20天年化波动30% -> 只拿 15/30 = 0.5 仓
        平静期波动10% -> 15/10=1.5，被 cap 到 1.0（不加杠杆）
    效果：市场发疯时自动减仓 -> 压回撤、稳Sharpe。
    """
    realized = returns.rolling(window).std() * np.sqrt(252)
    scale = (target_vol / realized).clip(upper=cap)
    return (position * scale).fillna(0.0)


def xs_momentum(prices: pd.DataFrame, lookback: int = 126,
                top_n: int = 2) -> pd.DataFrame:
    """横截面动量：每月末按过去 lookback 天收益给所有资产排名，
    下个月等权持有前 top_n 名。

    与单资产策略的本质区别：它比较的是"谁比谁强"（相对强弱），
    而不是"自己比过去强"（绝对趋势）。
    """
    mom = prices / prices.shift(lookback) - 1
    ranks = mom.rank(axis=1, ascending=False)          # 1 = 最强
    daily_w = (ranks <= top_n).astype(float)
    row_sum = daily_w.sum(axis=1)
    daily_w = daily_w.div(row_sum.where(row_sum > 0), axis=0).fillna(0.0)

    # 只在每月最后一个交易日"落子"，其余日子沿用上次的权重
    month = pd.Series(prices.index.month, index=prices.index)
    is_month_end = month != month.shift(-1)            # 下一行月份变了 -> 今天是月末
    weights = daily_w.where(is_month_end).ffill().fillna(0.0)
    return weights
