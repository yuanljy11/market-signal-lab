"""统计层：项目从"教程"变"研究"的地方。

包含：
1. block bootstrap —— 检验 "策略Sharpe - 基准Sharpe" 是否显著异于0
2. 参数敏感性网格 —— 回答"你是不是挑了个好看的参数"
3. walk-forward —— 滚动的样本外验证
4. 分市场环境表现 —— 牛市/崩盘/熊市分段
"""
import numpy as np
import pandas as pd

from backtest import backtest_single
from config import TRADING_DAYS
from metrics import sharpe
from signals import ma_crossover


def _sharpe_np(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return x.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan


def block_bootstrap_sharpe_diff(r_strat: pd.Series, r_bench: pd.Series,
                                block: int = 20, n_boot: int = 5000,
                                seed: int = 42) -> dict:
    """检验 H0: Sharpe(策略) == Sharpe(基准)。

    为什么是 *block* bootstrap 而不是逐日抽样：
    日收益不独立（波动聚集：大动之后往往接着大动）。逐日重抽会打碎
    这种自相关，低估方差 -> 假显著。整块(约1个月)抽样保留块内相关结构。

    做法：把两条收益序列"绑在一起"按同样的日期块重抽（保留两者的相关性），
    每次重算 Sharpe 差，重复 n_boot 次得到差值的抽样分布。
    """
    df = pd.concat([r_strat, r_bench], axis=1).dropna()
    a, b = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    n = len(a)
    rng = np.random.default_rng(seed)

    observed = _sharpe_np(a) - _sharpe_np(b)

    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        diffs[i] = _sharpe_np(a[idx]) - _sharpe_np(b[idx])

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # 双侧 bootstrap p 值：抽样分布落在0哪一侧的比例，取小的一侧 *2
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "observed_diff": round(float(observed), 3),
        "ci_95": (round(float(ci_low), 3), round(float(ci_high), 3)),
        "p_value": round(float(min(p, 1.0)), 4),
        "significant_5pct": bool(ci_low > 0 or ci_high < 0),
    }


def ma_param_grid(prices: pd.Series, returns: pd.Series,
                  fasts, slows, cost: float) -> pd.DataFrame:
    """对每组 (fast, slow) 算净Sharpe，产出热力图数据。
    读法：如果只有一格亮、周围全暗 -> 那格是运气/过拟合；
    整片区域都不错 -> 策略对参数稳健。"""
    grid = pd.DataFrame(index=fasts, columns=slows, dtype=float)
    for f in fasts:
        for s in slows:
            if f >= s:
                continue
            pos = ma_crossover(prices, f, s)
            res = backtest_single(returns, pos, cost)
            grid.loc[f, s] = sharpe(res["net"])
    grid.index.name = "fast MA"
    grid.columns.name = "slow MA"
    return grid.astype(float).round(2)


def walk_forward(prices: pd.Series, returns: pd.Series, param_pairs,
                 train_years: int = 3, test_years: int = 1,
                 cost: float = 0.001):
    """滚动样本外：每个窗口在训练段选净Sharpe最高的参数，
    只把这组参数用到紧接着的测试段。测试段收益拼起来 = 真实可得的
    样本外净值（每一天用的参数都只依赖它之前的数据）。

    注意：MA 在全序列上计算没有泄漏 —— rolling 只用过去数据；
    泄漏发生在"用未来数据选参数"，这正是本函数要杜绝的。
    """
    years = sorted(returns.index.year.unique())
    records, test_parts = [], []

    for start in years:
        train_mask = (returns.index.year >= start) & (returns.index.year < start + train_years)
        test_mask = (returns.index.year >= start + train_years) & (returns.index.year < start + train_years + test_years)
        if test_mask.sum() == 0:
            break

        best_params, best_sh = None, -np.inf
        for f, s in param_pairs:
            res = backtest_single(returns, ma_crossover(prices, f, s), cost)
            sh = sharpe(res["net"][train_mask])
            if not np.isnan(sh) and sh > best_sh:
                best_sh, best_params = sh, (f, s)

        res = backtest_single(returns, ma_crossover(prices, *best_params), cost)
        oos = res["net"][test_mask]
        test_parts.append(oos)
        records.append({
            "train": f"{start}-{start + train_years - 1}",
            "test": f"{start + train_years}",
            "chosen_params": best_params,
            "train_sharpe": round(best_sh, 2),
            "test_sharpe": round(sharpe(oos), 2) if len(oos) else np.nan,
        })

    oos_returns = pd.concat(test_parts).sort_index()
    return pd.DataFrame(records), oos_returns


REGIMES = {
    "2015-2019 (bull)": ("2015-01-01", "2019-12-31"),
    "2020 (covid crash+rebound)": ("2020-01-01", "2020-12-31"),
    "2021 (late bull)": ("2021-01-01", "2021-12-31"),
    "2022 (bear)": ("2022-01-01", "2022-12-31"),
    "2023-now": ("2023-01-01", None),
}


def regime_table(returns_dict: dict) -> pd.DataFrame:
    """各策略在不同市场环境下的Sharpe —— 回答"什么时候有效、什么时候失效"。"""
    rows = {}
    for regime, (lo, hi) in REGIMES.items():
        rows[regime] = {
            name: round(sharpe(r.loc[lo:hi]), 2) for name, r in returns_dict.items()
        }
    return pd.DataFrame(rows).T
