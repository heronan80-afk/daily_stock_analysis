# -*- coding: utf-8 -*-
"""本地筹码分布 (CYQ / 成本转换分布) 计算。

忠实移植东方财富前端 CYQCalculator 的等价逻辑（算法参考：
chengzuopeng/stock-sdk `src/indicators/chip.ts`，其注释明确该 TS 实现与
akshare `stock_cyq_em` 运行的是同一段东财 JS）。用日 K 的 OHLC + 换手率
在本地推演全体流通筹码的持仓成本分布，无需访问东财筹码接口，天然免疫
`ak.stock_cyq_em` 的断连问题。

模型要点：
- 价格档 FACTOR=150，窗口内 [min(low), max(high)] 均分，档宽精度下限 0.01 元；
- 逐日：先把存量筹码整体 ×(1-换手率)，再把当日换手筹码按三角形分布
  （顶点在均价 avg=(O+C+H+L)/4）铺到 [low, high]；一字板(high==low)全堆入单档；
- 口径：本模块按 range=0 全量累计（akshare stock_cyq_em 的口径），即窗口从
  传入序列首根起全量累计，只对最后一个交易日输出统计行；
- 从分布读出：获利比例、平均成本（累计 50% 筹码处价格，即中位数成本）、
  90/70 成本区间与集中度 (高-低)/(高+低)。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# 价格档数量（东财原版 factor = 150）
FACTOR = 150
# 比例类字段（获利比例 / 集中度）输出舍入小数位（原版默认 3 位）
_DECIMALS = 3
# 换手率字段缺失时用于换算的最小字段组合
_VOL_COL = "vol"
_FLOAT_SHARE_COL = "float_share"

_REQUIRED_COLS = ("date", "open", "high", "low", "close")


def _prec12(v: float) -> float:
    """原版 `Number(x.toPrecision(12))` 的等价写法（压浮点尾数）。"""
    return float(f"{v:.12g}")


def _to_price(v: float) -> float:
    """原版 `Number(v.toFixed(2))` 的等价写法（价格输出固定 2 位小数）。"""
    return round(v, 2)


def _round_dec(v: float, d: int = _DECIMALS) -> float:
    """比例类字段的十进制舍入（原版 round 语义，默认 3 位）。"""
    return round(v, d)


def _clamp_bucket(i: int) -> int:
    """档位索引夹逼到 [0, FACTOR-1]（原版 clamp，防御脏数据越界）。"""
    return max(0, min(FACTOR - 1, i))


def _resolve_turnover_fraction(df: pd.DataFrame) -> np.ndarray:
    """逐行解析换手比例（0..1）。

    优先级：
    1. `turnover_rate` 列（Tushare daily_basic 口径，百分数）→ /100；
    2. 无 turnover_rate 时用 `vol`(手)*100 / `float_share`(万股)*10000；
    3. 两者皆缺/非法 → 0（该 bar 当日不换手，沿用东财 `hsl/100 || 0` 语义）。
    """
    n = len(df)
    fractions = np.zeros(n, dtype=float)
    if "turnover_rate" in df.columns:
        tr = pd.to_numeric(df["turnover_rate"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(tr) & (tr > 0)
        fractions[valid] = tr[valid] / 100.0
        # turnover_rate 非法但可用 vol/float_share 兜底的行，走第二步换算
        need_fallback = ~valid
    else:
        need_fallback = np.ones(n, dtype=bool)
    if need_fallback.any() and _VOL_COL in df.columns and _FLOAT_SHARE_COL in df.columns:
        vol = pd.to_numeric(df[_VOL_COL], errors="coerce").to_numpy(dtype=float)
        float_share = pd.to_numeric(df[_FLOAT_SHARE_COL], errors="coerce").to_numpy(dtype=float)
        ok = need_fallback & np.isfinite(vol) & np.isfinite(float_share) & (vol > 0) & (float_share > 0)
        fractions[ok] = (vol[ok] * 100.0) / (float_share[ok] * 10000.0)
    # 换手率夹逼到 [0,1]（脏数据负换手会让衰减因子>1，夹到 0 = 当日不换手）
    return np.clip(fractions, 0.0, 1.0)


def _valid_mask(df: pd.DataFrame) -> np.ndarray:
    """OHLC 全部有限的行才参与分布推演（上游脏行跳过，不中断整段计算）。"""
    mask = np.ones(len(df), dtype=bool)
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            mask &= np.isfinite(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
        else:
            mask &= False
    return mask


def calc_cyq_metrics(daily_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """计算最后一个交易日的筹码分布统计（range=0 全量累计，akshare 口径）。

    Args:
        daily_df: 按时间升序的日 K DataFrame，至少含
            `date/open/high/low/close`；换手率可选 `turnover_rate`(百分数)，
            或用 `vol`(手)+`float_share`(万股) 兜底换算。

    Returns:
        与 `ChipDistribution` 字段对齐的 dict：
        ``date/profit_ratio/avg_cost/cost_90_low/cost_90_high/concentration_90/
        cost_70_low/cost_70_high/concentration_70``（比例类为 0..1 小数）。
        分布退化（窗口内换手全为 0/缺失，总筹码为 0）或无有效 bar 时返回 None。
    """
    if daily_df is None or daily_df.empty:
        return None
    for col in _REQUIRED_COLS:
        if col not in daily_df.columns:
            return None

    n = len(daily_df)
    valid = _valid_mask(daily_df)
    if not valid.any():
        return None

    highs = pd.to_numeric(daily_df["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(daily_df["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(daily_df["close"], errors="coerce").to_numpy(dtype=float)
    opens = pd.to_numeric(daily_df["open"], errors="coerce").to_numpy(dtype=float)
    turnovers = _resolve_turnover_fraction(daily_df)

    # 窗口价格域（仅有效 bar 参与 min/max）
    maxprice = float(np.max(highs[valid]))
    minprice = float(np.min(lows[valid]))

    # 档宽精度不小于 0.01（产品逻辑，与东财一致）
    accuracy = max(0.01, (maxprice - minprice) / (FACTOR - 1))

    # 筹码堆叠：逐 bar 衰减 + 三角形分布叠加
    xdata = np.zeros(FACTOR, dtype=float)
    for i in range(n):
        if not valid[i]:
            continue
        o, c, h, low = opens[i], closes[i], highs[i], lows[i]
        turnover = turnovers[i]
        avg = (o + c + h + low) / 4.0

        H = _clamp_bucket(int(math.floor((h - minprice) / accuracy)))
        L = _clamp_bucket(int(math.ceil((low - minprice) / accuracy)))
        # G 点：一字板时 X 为进度因子（矩形面积是三角形的 2 倍）
        g_factor = FACTOR - 1 if h == low else 2.0 / (h - low)
        g_index = _clamp_bucket(int(math.floor((avg - minprice) / accuracy)))

        # 衰减：当日换手部分从存量筹码中等比例移除
        xdata *= 1.0 - turnover

        if h == low:
            xdata[g_index] += (g_factor * turnover) / 2.0
            continue

        bucket_idx = np.arange(L, H + 1, dtype=int)
        prices = minprice + accuracy * bucket_idx
        upper = prices <= avg
        weight = np.zeros(len(bucket_idx), dtype=float)
        if upper.any():
            if abs(avg - low) < 1e-8:
                weight[upper] = 1.0
            else:
                weight[upper] = (prices[upper] - low) / (avg - low)
        if (~upper).any():
            if abs(h - avg) < 1e-8:
                weight[~upper] = 1.0
            else:
                weight[~upper] = (h - prices[~upper]) / (h - avg)
        xdata[bucket_idx] += weight * g_factor * turnover

    # 读出阶段统一 pre-tabulate 归一化值（原版 x.toPrecision(12)）
    xp = np.array([_prec12(float(v)) for v in xdata], dtype=float)
    total_chips = float(xp.sum())
    # 分布退化（窗口内换手全为 0/缺失）：原版输出全 0，这里输出 None 更诚实
    if total_chips == 0:
        return None

    def get_cost_by_chip(chip: float) -> float:
        """累计到指定筹码量处的成本价（原版 getCostByChip）。"""
        cost = 0.0
        acc = 0.0
        for i in range(FACTOR):
            x = xp[i]
            if acc + x > chip:
                cost = minprice + i * accuracy
                break
            acc += x
        return cost

    def get_benefit_part(price: float) -> float:
        """指定价格的获利比例（原版 getBenefitPart）。"""
        below = 0.0
        for i in range(FACTOR):
            if price >= minprice + i * accuracy:
                below += xp[i]
        return below / total_chips

    def compute_percent_chips(percent: float) -> Dict[str, float]:
        """中间 percent 筹码的价格区间与集中度（原版 computePercentChips）。"""
        pr_low = get_cost_by_chip(total_chips * ((1.0 - percent) / 2.0))
        pr_high = get_cost_by_chip(total_chips * ((1.0 + percent) / 2.0))
        if pr_low + pr_high == 0:
            concentration = 0.0
        else:
            concentration = (pr_high - pr_low) / (pr_low + pr_high)
        return {
            "low": _to_price(pr_low),
            "high": _to_price(pr_high),
            "concentration": concentration,
        }

    last_idx = int(np.where(valid)[0][-1])
    close = closes[last_idx]
    p90 = compute_percent_chips(0.9)
    p70 = compute_percent_chips(0.7)

    return {
        "date": str(daily_df.iloc[last_idx]["date"]),
        "profit_ratio": _round_dec(get_benefit_part(close)),
        "avg_cost": _to_price(get_cost_by_chip(total_chips * 0.5)),
        "cost_90_low": p90["low"],
        "cost_90_high": p90["high"],
        "concentration_90": _round_dec(p90["concentration"]),
        "cost_70_low": p70["low"],
        "cost_70_high": p70["high"],
        "concentration_70": _round_dec(p70["concentration"]),
    }
