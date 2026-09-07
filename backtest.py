
import pandas as pd


def backtest_single(returns: pd.Series, position: pd.Series,
                    cost: float = 0.001) -> pd.DataFrame:
   
    pos = position.shift(1).fillna(0.0)       
    gross = pos * returns
    trades = pos.diff().abs().fillna(0.0)      
    net = gross - trades * cost
    return pd.DataFrame({"gross": gross, "net": net, "trade": trades})


def backtest_portfolio(returns: pd.DataFrame, weights: pd.DataFrame,
                       cost: float = 0.001) -> pd.DataFrame:

    w = weights.shift(1).fillna(0.0)
    gross = (w * returns).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * cost
    return pd.DataFrame({"gross": gross, "net": net, "trade": turnover})
