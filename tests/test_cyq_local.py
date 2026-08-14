# -*- coding: utf-8 -*-
"""本地 CYQ 筹码分布算法（src/services/cyq.py）纯函数测试。

算法忠实移植东方财富 CYQCalculator（等价 akshare stock_cyq_em 的 JS 逻辑），
测试用手算可验证的合成 K 线钉住关键语义：一字板、三角分布、衰减累积、
获利比例边界、分布退化、脏 bar 跳过、换手率兜底换算。
"""

import pandas as pd
import pytest

from src.services.cyq import calc_cyq_metrics


def _df(rows):
    return pd.DataFrame(rows)


def test_one_word_board_accumulates_all_chips_at_single_price():
    """一字板（high==low）：全部筹码堆入单一价格档 → 集中度 0、获利比例 1。"""
    df = _df([
        {"date": "2026-08-14", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
         "turnover_rate": 100.0},
    ])
    r = calc_cyq_metrics(df)
    assert r is not None
    assert r["date"] == "2026-08-14"
    assert r["profit_ratio"] == pytest.approx(1.0)
    assert r["avg_cost"] == pytest.approx(10.0)
    assert r["cost_90_low"] == pytest.approx(10.0)
    assert r["cost_90_high"] == pytest.approx(10.0)
    assert r["concentration_90"] == pytest.approx(0.0)
    assert r["concentration_70"] == pytest.approx(0.0)


def test_two_day_decay_moves_median_cost_to_newer_price():
    """衰减累积：第 1 天满换手在 10 元，第 2 天 50% 换手在 20 元 →
    50% 中位数成本落在 20 元档（旧筹码被衰减一半）。"""
    df = _df([
        {"date": "2026-08-13", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
         "turnover_rate": 100.0},
        {"date": "2026-08-14", "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0,
         "turnover_rate": 50.0},
    ])
    r = calc_cyq_metrics(df)
    assert r is not None
    assert r["date"] == "2026-08-14"
    # 总筹码：10 元档 37.25（74.5 衰减一半）+ 20 元档 37.25 → 中位数在 20 元档
    assert r["avg_cost"] == pytest.approx(20.0)
    assert r["profit_ratio"] == pytest.approx(1.0)


def test_triangle_distribution_symmetric_bands():
    """对称三角形分布（O=8 H=12 L=8 C=12，峰值在 avg=10）：
    90% 成本区间应大致对称于 10，集中度 ~0.13。"""
    df = _df([
        {"date": "2026-08-14", "open": 8.0, "high": 12.0, "low": 8.0, "close": 12.0,
         "turnover_rate": 100.0},
    ])
    r = calc_cyq_metrics(df)
    assert r is not None
    assert r["profit_ratio"] == pytest.approx(1.0)  # 收盘=最高价 → 全部获利
    assert r["avg_cost"] == pytest.approx(10.0, abs=0.1)
    assert r["cost_90_low"] == pytest.approx(8.6, abs=0.3)
    assert r["cost_90_high"] == pytest.approx(11.4, abs=0.3)
    assert r["concentration_90"] == pytest.approx(0.136, abs=0.02)
    assert r["cost_70_low"] > r["cost_90_low"]
    assert r["cost_70_high"] < r["cost_90_high"]


def test_profit_ratio_below_all_cost_returns_zero():
    """收盘价低于所有筹码档 → 获利比例 0。"""
    # 收盘 5 元，但筹码堆在 10 元（最高/最低都是 10 的一字板后接一根正常 K）
    df = _df([
        {"date": "2026-08-13", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
         "turnover_rate": 100.0},
        {"date": "2026-08-14", "open": 5.0, "high": 5.0, "low": 5.0, "close": 5.0,
         "turnover_rate": 0.0},
    ])
    r = calc_cyq_metrics(df)
    assert r is not None
    assert r["profit_ratio"] == pytest.approx(0.0)


def test_zero_turnover_degenerates_to_none():
    """窗口内换手全为 0 → 总筹码为 0 → 返回 None（拒绝全 0 分布）。"""
    df = _df([
        {"date": "2026-08-14", "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.2,
         "turnover_rate": 0.0},
    ])
    assert calc_cyq_metrics(df) is None


def test_negative_turnover_clamped_to_zero():
    """负换手率被夹到 0（当日不换手），不导致衰减因子 >1。"""
    df = _df([
        {"date": "2026-08-14", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
         "turnover_rate": -50.0},
    ])
    # 换手 0 → 退化 → None（不会被负换手放大成非法分布）
    assert calc_cyq_metrics(df) is None


def test_dirty_bars_skipped_and_last_valid_date_used():
    """OHLC 含空值/NaN 的脏行被跳过，不中断整段计算，日期取最后一个有效 bar。"""
    df = _df([
        {"date": "2026-08-13", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
         "turnover_rate": 100.0},
        {"date": "2026-08-14", "open": None, "high": None, "low": None, "close": None,
         "turnover_rate": 100.0},
    ])
    r = calc_cyq_metrics(df)
    assert r is not None
    assert r["date"] == "2026-08-13"
    assert r["avg_cost"] == pytest.approx(10.0)


def test_turnover_fallback_from_vol_and_float_share():
    """无 turnover_rate 时用 vol(手)*100 / float_share(万股)*10000 换算换手比例。"""
    # vol=1000 手 → 100000 股；float_share=100000 万股 → 1e9 股 → 换手 1e-4
    df = _df([
        {"date": "2026-08-14", "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
         "vol": 1000.0, "float_share": 100000.0},
    ])
    r = calc_cyq_metrics(df)
    assert r is not None
    # 换手极低（1e-4）但非零 → 分布不退化为 None
    assert r["avg_cost"] == pytest.approx(10.0)


def test_missing_required_columns_returns_none():
    df = _df([
        {"date": "2026-08-14", "open": 10.0, "high": 10.0, "low": 10.0},  # 缺 close
    ])
    assert calc_cyq_metrics(df) is None
    assert calc_cyq_metrics(pd.DataFrame()) is None
    assert calc_cyq_metrics(None) is None
