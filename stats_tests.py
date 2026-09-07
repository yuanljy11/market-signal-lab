
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

    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "observed_diff": round(float(observed), 3),
        "ci_95": (round(float(ci_low), 3), round(float(ci_high), 3)),
        "p_value": round(float(min(p, 1.0)), 4),
        "significant_5pct": bool(ci_low > 0 or ci_high < 0),
    }


def ma_param_grid(prices: pd.Series, returns: pd.Series,
                  fasts, slows, cost: float) -> pd.DataFrame:

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

    rows = {}
    for regime, (lo, hi) in REGIMES.items():
        rows[regime] = {
            name: round(sharpe(r.loc[lo:hi]), 2) for name, r in returns_dict.items()
        }
    return pd.DataFrame(rows).T
