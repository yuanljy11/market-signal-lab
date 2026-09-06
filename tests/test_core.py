"""单元测试：用手算得出答案的迷你数据验证核心逻辑。

面试话术：'我的回测引擎有测试覆盖，包括一个专门证明
没有未来函数的测试' —— 这句话大部分候选人说不出来。
运行：pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from backtest import backtest_portfolio, backtest_single
from metrics import max_drawdown, sharpe
from signals import vol_target


def _dates(n):
    return pd.bdate_range("2020-01-01", periods=n)


def test_no_lookahead():
    """信号只在最后一天变成1，而最后一天有+50%的暴涨。
    如果引擎正确(shift(1))，策略吃不到这口肉 -> 总收益必须是0。
    如果有人删掉shift(1)，这个测试立刻爆红。"""
    returns = pd.Series([0.0, 0.0, 0.0, 0.5], index=_dates(4))
    position = pd.Series([0, 0, 0, 1], index=_dates(4), dtype=float)
    res = backtest_single(returns, position, cost=0.0)
    assert res["gross"].sum() == 0.0


def test_signal_earns_next_day():
    """Day0 发出信号 -> 吃到 Day1 的收益。"""
    returns = pd.Series([0.0, 0.02, 0.0], index=_dates(3))
    position = pd.Series([1, 0, 0], index=_dates(3), dtype=float)
    res = backtest_single(returns, position, cost=0.0)
    assert res["gross"].iloc[1] == pytest.approx(0.02)


def test_cost_charged_on_change():
    """仓位 0->1->1->0：只有两次变化，各收一次成本。"""
    returns = pd.Series([0.0] * 5, index=_dates(5))
    position = pd.Series([0, 1, 1, 0, 0], index=_dates(5), dtype=float)
    res = backtest_single(returns, position, cost=0.001)
    assert res["trade"].sum() == pytest.approx(2.0)
    assert res["net"].sum() == pytest.approx(-0.002)


def test_max_drawdown_hand_computed():
    """财富 1.0 -> 1.1 -> 0.99 -> 1.05，手算最大回撤 = 0.99/1.1 - 1 = -10%。"""
    returns = pd.Series([0.10, -0.10, 0.0606060606], index=_dates(3))
    assert max_drawdown(returns) == pytest.approx(-0.10, abs=1e-6)


def test_sharpe_scaling():
    """日均0.05%、日波动1% -> 年化Sharpe约0.79（√252年化）。"""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 100_000))
    assert sharpe(r) == pytest.approx(0.0005 / 0.01 * np.sqrt(252), rel=0.05)


def test_vol_target_reduces_position_in_high_vol():
    """波动30% > 目标15% -> 仓位应缩到约0.5；且不超过cap。"""
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.30 / np.sqrt(252), 300), index=_dates(300))
    pos = pd.Series(1.0, index=_dates(300))
    scaled = vol_target(pos, r, target_vol=0.15, window=20, cap=1.0)
    assert scaled.iloc[50:].mean() == pytest.approx(0.5, abs=0.15)
    assert scaled.max() <= 1.0


def test_portfolio_weights_shifted():
    """组合版同样必须无未来函数。"""
    idx = _dates(3)
    returns = pd.DataFrame({"A": [0.0, 0.0, 0.4], "B": [0.0, 0.0, 0.0]}, index=idx)
    weights = pd.DataFrame({"A": [0, 0, 1], "B": [0, 0, 0]}, index=idx, dtype=float)
    res = backtest_portfolio(returns, weights, cost=0.0)
    assert res["gross"].sum() == 0.0
