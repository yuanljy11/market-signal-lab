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
<!-- 跑完 run_analysis.py 后把 reports/ 里的表和图贴进来，如实描述 -->

## Key Findings
<!-- 敢写不显著：e.g. "Sharpe improvement is not significant at 5% (CI [-0.05, 0.48])" -->

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
