# Market Signal Lab: Multi-Strategy Trading Research & Risk Analysis

## Research Question
> Can simple momentum-based trading signals generate superior **risk-adjusted** returns
> compared with buy-and-hold across asset classes — and does the edge survive
> transaction costs, volatility targeting, and statistical significance testing?

## Data
Daily adjusted close for SPY, QQQ, IWM, GLD, TLT (2015–present, Yahoo Finance, cached in `data/`).

## Strategies
| Strategy | Type | Idea |
|---|---|---|
| MA 20/100 crossover | trend, single-asset | baseline |
| TSMOM (12m) | time-series momentum | Moskowitz, Ooi & Pedersen (2012) |
| TSMOM + Vol Targeting | risk-managed | position = target_vol / realized_vol |
| XS Momentum (top-2 of 5) | cross-sectional, portfolio | monthly rebalance |

## Methodology
1. Signals use information up to day *t*; positions applied from day *t+1* (`shift(1)` — verified by unit test)
2. 0.1% one-way transaction cost on position changes
3. Parameter sensitivity grid (MA fast × slow)
4. Walk-forward validation (3y train / 1y test, rolling)
5. **Block bootstrap** (block=20d, n=5000) test of Sharpe difference vs benchmark
6. Regime analysis: bull / covid crash / bear sub-periods

## Results

Over 2015–2026, volatility-targeted time-series momentum (TSMOM + VolTarget) matched
buy-and-hold on risk-adjusted returns for US large caps (SPY Sharpe 0.84 vs 0.83; QQQ
0.89 vs 0.90) while cutting risk roughly in half: annualised volatility fell from 17.6%
to 11.0% and maximum drawdown from −33.7% to −17.3% on SPY (QQQ: −35.1% to −17.5%),
lifting Calmar from 0.41 to 0.52. Results varied sharply by asset class: trend-following
was the only approach to deliver positive risk-adjusted returns on long-duration
Treasuries during the 2022–2024 rate-hike bear market (TLT MA-crossover Sharpe 0.23 vs
0.01 for buy-and-hold), but underperformed on small caps (IWM) and gold. Cross-sectional
momentum (top-2 of 5, monthly) failed to beat an equal-weight benchmark (Sharpe 0.73 vs
0.93). Parameter sensitivity was moderate (net Sharpe 0.44–0.90 across the MA grid, no
isolated optimum), but walk-forward validation showed unstable parameter selection, with
out-of-sample Sharpe ranging from −2.21 (2022) to +1.88 (2021).

## Key Findings

1. **No statistically significant alpha.** The Sharpe improvement of the risk-managed
   momentum strategy over buy-and-hold on SPY was 0.01 (block bootstrap, n=5000,
   block=20d; 95% CI [−0.36, 0.37]; p = 0.97) — consistent with the efficient market
   hypothesis for simple rules on public information.
2. **The value of the strategy is risk reduction, not return enhancement.** It halved
   maximum drawdown at comparable Sharpe — effectively a cheaper ride to a similar
   destination, which matters because deep drawdowns compound asymmetrically.
3. **Trend-following behaves like insurance with regime-dependent premiums.** It
   protected capital in the slow 2022–2024 bond bear (TLT) but was whipsawed by 2022's
   bear-market rallies in equities, where its within-year Sharpe was worse than
   buy-and-hold even as full-period drawdown stayed lower.
4. **Walk-forward analysis reveals overfitting risk**: parameters chosen on 3-year
   training windows generalised poorly to the following year, cautioning against
   in-sample parameter optimisation.


## Limitations
- Risk-free rate assumed 0 in Sharpe; no slippage model beyond flat 0.1% cost
- √252 annualisation assumes i.i.d. daily returns (ignores volatility clustering)
- Survivorship-free broad ETFs only; results may not generalise to single stocks

## Reproduce
```bash
pip install -r requirements.txt
pytest tests/ -v        # 7 tests incl. look-ahead-bias guard
python run_analysis.py  # all tables & figures -> reports/
```
