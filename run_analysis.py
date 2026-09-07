
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as C
from backtest import backtest_portfolio, backtest_single
from data import load_prices, sanity_check
from metrics import summary
from signals import ma_crossover, tsmom, vol_target, xs_momentum
from stats_tests import (block_bootstrap_sharpe_diff, ma_param_grid,
                         regime_table, walk_forward)

OUT = Path("reports")
OUT.mkdir(exist_ok=True)


def strategies_for_asset(prices: pd.Series, returns: pd.Series) -> dict:

    ma_pos = ma_crossover(prices, 20, 100)
    ts_pos = tsmom(prices, C.TSMOM_LOOKBACK)
    vt_pos = vol_target(ts_pos, returns, C.VOL_TARGET, C.VOL_WINDOW, C.VOL_CAP)

    return {
        "Buy&Hold": returns,
        "MA 20/100": backtest_single(returns, ma_pos, C.COST)["net"],
        "TSMOM 12m": backtest_single(returns, ts_pos, C.COST)["net"],
        "TSMOM + VolTarget": backtest_single(returns, vt_pos, C.COST)["net"],
    }


def main():
    prices = load_prices(C.TICKERS, C.START)
    sanity_check(prices)
    returns = prices / prices.shift(1) - 1

    all_results = {}
    for t in C.TICKERS:
        strat_rets = strategies_for_asset(prices[t], returns[t])
        all_results[t] = strat_rets
        tbl = summary(strat_rets, C.RF)
        tbl.to_csv(OUT / f"summary_{t}.csv")
        print(f"\n===== {t} =====\n{tbl}")

        wealth = pd.DataFrame({k: (1 + v.fillna(0)).cumprod() for k, v in strat_rets.items()})
        wealth.plot(figsize=(11, 5), title=f"{t}: Growth of $1 (net of costs)")
        plt.ylabel("Wealth")
        plt.tight_layout()
        plt.savefig(OUT / f"wealth_{t}.png", dpi=150)
        plt.close()


    xs_w = xs_momentum(prices, C.XS_LOOKBACK, C.XS_TOP_N)
    xs_net = backtest_portfolio(returns, xs_w, C.COST)["net"]
    ew_bench = returns.mean(axis=1)  # 等权持有5资产作为组合基准
    xs_tbl = summary({"EqualWeight B&H": ew_bench, "XS Momentum top2": xs_net}, C.RF)
    xs_tbl.to_csv(OUT / "summary_xs_momentum.csv")
    print(f"\n===== Cross-sectional momentum (portfolio) =====\n{xs_tbl}")

    spy_p, spy_r = prices["SPY"], returns["SPY"]

    grid = ma_param_grid(spy_p, spy_r, C.MA_FASTS, C.MA_SLOWS, C.COST)
    grid.to_csv(OUT / "spy_ma_param_grid.csv")
    print(f"\n===== SPY MA param sensitivity (net Sharpe) =====\n{grid}")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid.values.astype(float), cmap="RdYlGn")
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xlabel("slow MA"); ax.set_ylabel("fast MA")
    ax.set_title("SPY net Sharpe by MA params")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(OUT / "spy_ma_heatmap.png", dpi=150)
    plt.close()

    pairs = [(f, s) for f in C.MA_FASTS for s in C.MA_SLOWS if f < s]
    wf_tbl, wf_oos = walk_forward(spy_p, spy_r, pairs, cost=C.COST)
    wf_tbl.to_csv(OUT / "spy_walk_forward.csv", index=False)
    print(f"\n===== SPY walk-forward =====\n{wf_tbl}")

    boot = block_bootstrap_sharpe_diff(
        all_results["SPY"]["TSMOM + VolTarget"], spy_r,
        C.BOOT_BLOCK, C.BOOT_N, C.BOOT_SEED)
    print(f"\n===== Bootstrap: (TSMOM+VolTarget) Sharpe - (Buy&Hold) Sharpe, SPY =====\n{boot}")
    pd.Series(boot).to_csv(OUT / "spy_bootstrap.csv")

    reg = regime_table(all_results["SPY"])
    reg.to_csv(OUT / "spy_regimes.csv")
    print(f"\n===== SPY Sharpe by market regime =====\n{reg}")

    print(f"\nAll outputs saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()
