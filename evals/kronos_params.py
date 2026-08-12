"""Per-stock Kronos parameters based on sweep results.

Each stock can have different (lookback, T, top_p, sample_count) based on
which parameters performed best in the sweep.

用法:
    from evals.kronos_params import get_stock_params
    lookback, T, top_p, sample_count = get_stock_params("300308")
"""
from __future__ import annotations

from typing import Tuple

# 默认参数（lookback=200 在大多数股票上 MAPE 和偏误显著改善）
DEFAULT_PARAMS = (200, 0.5, 0.9, 1)  # lookback, T, top_p, sample_count

# 个股参数覆盖
# 规则：lookback=200 为默认，回退的股票用 lookback=400
# 来源: evals/results/sweep_best/kronos_summary_20260801_180521.md
STOCK_PARAMS: dict[str, Tuple[int, float, float, int]] = {
    # ---- lookback=200 趋势方向好或改善 ----
    "002463": (200, 0.5, 0.9, 1),  # 沪电股份 — 趋势 15/15 ✅
    "002916": (200, 0.5, 0.9, 1),  # 深南电路 — 趋势 14/16 ✅
    "300031": (200, 0.5, 0.9, 1),  # 宝通科技 — 趋势 12/15 (改善!) ✅
    "300052": (200, 0.5, 0.9, 1),  # 中青宝 — 趋势 7/16 (持平)
    "300308": (200, 0.5, 0.9, 1),  # 中际旭创 — 趋势 15/15 ✅
    "300418": (200, 0.5, 0.9, 1),  # 昆仑万维 — 趋势 10/15 (改善!) ✅
    "300499": (200, 0.5, 0.9, 1),  # 高澜股份 — 趋势 6/14 (持平)
    "300502": (200, 0.5, 0.9, 1),  # 新易盛 — 趋势 15/15 ✅
    "301018": (200, 0.5, 0.9, 1),  # 申菱环境 — 趋势 13/16 ✅
    "600183": (200, 0.5, 0.9, 1),  # 生益科技 — 趋势 15/15 ✅
    "603019": (200, 0.5, 0.9, 1),  # 中科曙光 — 趋势 7/14 (改善!) ✅
    "688041": (200, 0.5, 0.9, 1),  # 海光信息 — 趋势 15/15 ✅
    "002837": (200, 0.5, 0.9, 1),  # 英维克 — 趋势 5/13 (持平)

    # ---- lookback=400 趋势方向更好（lookback=200 回退） ----
    "300394": (400, 1.0, 0.9, 1),  # 天孚通信 — 趋势 15/15 (lb=200 时仅 7/15)
    "300467": (400, 1.0, 0.9, 1),  # 迅游科技 — 趋势 11/16 (lb=200 时仅 4/16)
}


def get_stock_params(code: str) -> Tuple[int, float, float, int]:
    """获取个股的 Kronos 参数。

    Args:
        code: 6 位股票代码，如 "300308"

    Returns:
        (lookback, T, top_p, sample_count)
        未配置的股票返回 DEFAULT_PARAMS。
    """
    return STOCK_PARAMS.get(code, DEFAULT_PARAMS)