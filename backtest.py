"""回测引擎：把"仓位"变成"策略收益"。全项目防作弊的唯一关口。

核心纪律（未来函数防线）：
    今天收盘算出的仓位，最早明天才能持有 -> position.shift(1)
所有策略统一在这里 shift，信号层不 shift，两边都 shift 或都不 shift
的bug就不会出现。
"""
import pandas as pd


def backtest_single(returns: pd.Series, position: pd.Series,
                    cost: float = 0.001) -> pd.DataFrame:
    """单资产回测。
    returns : 该资产的日收益
    position: 目标仓位（0/1 或 vol targeting 后的小数）
    cost    : 单边成本，按 |仓位变化| 收取（买卖各算一次变化）
    """
    pos = position.shift(1).fillna(0.0)        # <- 防未来函数的那一行
    gross = pos * returns
    trades = pos.diff().abs().fillna(0.0)      # 今天仓位改了多少就交易了多少
    net = gross - trades * cost
    return pd.DataFrame({"gross": gross, "net": net, "trade": trades})


def backtest_portfolio(returns: pd.DataFrame, weights: pd.DataFrame,
                       cost: float = 0.001) -> pd.DataFrame:
    """组合回测（给横截面动量用）。
    组合收益 = sum(昨天定的权重 * 今天各资产收益)
    换手率   = sum(|各资产权重变化|)，乘成本
    """
    w = weights.shift(1).fillna(0.0)
    gross = (w * returns).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * cost
    return pd.DataFrame({"gross": gross, "net": net, "trade": turnover})
