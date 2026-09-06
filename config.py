"""全局配置：所有"可调的决定"集中在这里，方便做敏感性分析时批量改。"""

TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
START = "2015-01-01"

TRADING_DAYS = 252          # 一年约252个交易日
COST = 0.001                # 单边交易成本 0.1%
RF = 0.0                    # 无风险利率，第一版设0，README的Limitations里说明

# MA 参数网格（敏感性热力图 + walk-forward 用）
MA_FASTS = [10, 20, 50]
MA_SLOWS = [60, 100, 200]

# 时序动量回看期（约12个月）
TSMOM_LOOKBACK = 252

# 波动率目标化
VOL_TARGET = 0.15           # 目标年化波动率 15%
VOL_WINDOW = 20             # 用最近20天估计实际波动
VOL_CAP = 1.0               # 仓位上限（不加杠杆）

# 横截面动量
XS_LOOKBACK = 126           # 过去6个月收益排名
XS_TOP_N = 2                # 持有前2名

# 样本内 / 样本外切分
TRAIN_END = "2021-12-31"

# Bootstrap
BOOT_BLOCK = 20             # block bootstrap 块长（约1个月）
BOOT_N = 5000
BOOT_SEED = 42
