#!/usr/bin/env python3
"""
Kronos-base vs Kronos-small 批量回测脚本

对比两个模型在上周交易日（2026-07-27 ~ 2026-07-31）对 15 只追踪股票的预测精度。

执行方式：
    cd daily_stock_analysis
    source venv/bin/activate
    python evals/kronos_base_compare.py

输出：
    evals/results/kronos_base_compare_<ts>.md  — 对比报告
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_provider import DataFetcherManager  # noqa: E402

logger = logging.getLogger("kronos_base_compare")


# ── 配置 ──────────────────────────────────────────────────────────────

# 预测基准日
# 一、上周交易日（已验证窗口有限，但用户指定）
PREDICTION_DATES_LAST_WEEK = [
    "2026-07-27",
    "2026-07-28",
    "2026-07-29",
    "2026-07-30",
    "2026-07-31",
]

# 二、全部历史报告日期（15 日窗口已完全走完，数据更充分）
PREDICTION_DATES_FULL = [
    "2026-07-06",
    "2026-07-07",
    "2026-07-08",
    "2026-07-09",
    "2026-07-10",
    "2026-07-13",
    "2026-07-14",
    "2026-07-15",
    "2026-07-16",
    "2026-07-17",
    "2026-07-20",
    "2026-07-21",
    "2026-07-22",
    "2026-07-23",
]

# 追踪股票（与 kronos_service.py 一致）
TRACKED_STOCKS: List[Tuple[str, str]] = [
    ("002463", "沪电股份"),
    ("002916", "深南电路"),
    ("300031", "宝通科技"),
    ("300052", "中青宝"),
    ("300308", "中际旭创"),
    ("300418", "昆仑万维"),
    ("300499", "高澜股份"),
    ("300502", "新易盛"),
    ("301018", "申菱环境"),
    ("600183", "生益科技"),
    ("603019", "中科曙光"),
    ("688041", "海光信息"),
    ("002837", "英维克"),
    ("300394", "天孚通信"),
    ("300467", "迅游科技"),
]

# 预测参数
DEFAULT_LOOKBACK = 200
DEFAULT_T = 0.5
DEFAULT_TOP_P = 0.9
DEFAULT_SAMPLE_COUNT = 1
PRED_LEN = 15
MAX_WORKERS = 5

# 置信度（来自回测结果）
BEARISH_CONFIDENCE = 91.1
BULLISH_CONFIDENCE = 78.2


# ── 模型加载 ──────────────────────────────────────────────────────────

_KRONOS_PATH = os.environ.get("KRONOS_PATH", "/tmp/Kronos")

_PREDICTOR_CACHE: dict = {}


def _load_kronos(model_name: str, device: str = "cpu"):
    """加载指定 Kronos 模型（带缓存）。"""
    cache_key = f"{model_name}:{device}"
    if cache_key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[cache_key]

    if _KRONOS_PATH not in sys.path:
        sys.path.insert(0, _KRONOS_PATH)

    import torch
    from model import Kronos, KronosTokenizer, KronosPredictor

    hf_name = f"NeoQuasar/{model_name}"
    hf_tokenizer = "NeoQuasar/Kronos-Tokenizer-base"

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    logger.info("加载模型: %s (device=%s)", hf_name, device)
    t0 = time.time()

    tokenizer = KronosTokenizer.from_pretrained(hf_tokenizer, token=False)
    model = Kronos.from_pretrained(hf_name, token=False)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("  %s: %.1fM params, 加载耗时: %.1fs", model_name, params, time.time() - t0)

    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    _PREDICTOR_CACHE[cache_key] = predictor
    return predictor


# ── 数据 ──────────────────────────────────────────────────────────────

def get_stock_params(code: str) -> Tuple[int, float, float, int]:
    """获取个股参数（复用 kronos_params）。"""
    try:
        sys.path.insert(0, str(ROOT))
        from evals.kronos_params import get_stock_params
        return get_stock_params(code)
    except Exception:
        return (DEFAULT_LOOKBACK, DEFAULT_T, DEFAULT_TOP_P, DEFAULT_SAMPLE_COUNT)


# ── 预测结果 ──────────────────────────────────────────────────────────

@dataclass
class KronosPredResult:
    """单个 Kronos 预测结果。"""
    stock_code: str
    stock_name: str
    report_date: str
    model_name: str
    available: bool
    direction: str          # 看多 / 看空 / 震荡
    pred_pct_change: float  # 预测区间总涨跌幅
    actual_pct_change: float  # 实际区间总涨跌幅
    actual_direction: str   # 实际方向
    hit: bool               # 方向是否命中
    pred_days: int          # 实际可用的预测天数
    pred_days_requested: int  # 请求的预测天数
    error: str              # 失败原因
    lookback: int
    window_too_short: bool = False  # 验证窗口不足（实际交易日 < 5）
    raw_pred_close: List[float] = field(default_factory=list)
    raw_actual_close: List[float] = field(default_factory=list)


def run_prediction(
    stock_code: str,
    stock_name: str,
    report_date: str,
    model_name: str,
    fetcher_mgr: DataFetcherManager,
    pred_len: int = PRED_LEN,
) -> KronosPredResult:
    """对单只股票在指定日期运行 Kronos 预测。"""
    predictor = _load_kronos(model_name)
    if predictor is None:
        return KronosPredResult(
            stock_code=stock_code, stock_name=stock_name,
            report_date=report_date, model_name=model_name,
            available=False, direction="", pred_pct_change=0,
            actual_pct_change=0, actual_direction="", hit=False,
            pred_days=0, pred_days_requested=pred_len,
            error="模型加载失败", lookback=0,
        )

    lookback, T, top_p, sample_count = get_stock_params(stock_code)

    # 获取行情
    start_dt = pd.to_datetime(report_date) - timedelta(days=lookback * 2)
    end_dt = pd.to_datetime(report_date) + timedelta(days=pred_len * 2)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    try:
        df, source = fetcher_mgr.get_daily_data(stock_code, start_date=start_str, end_date=end_str)
    except Exception as e:
        return KronosPredResult(
            stock_code=stock_code, stock_name=stock_name,
            report_date=report_date, model_name=model_name,
            available=False, direction="", pred_pct_change=0,
            actual_pct_change=0, actual_direction="", hit=False,
            pred_days=0, pred_days_requested=pred_len,
            error=f"获取行情失败: {e}", lookback=lookback,
        )

    if df is None or df.empty:
        return KronosPredResult(
            stock_code=stock_code, stock_name=stock_name,
            report_date=report_date, model_name=model_name,
            available=False, direction="", pred_pct_change=0,
            actual_pct_change=0, actual_direction="", hit=False,
            pred_days=0, pred_days_requested=pred_len,
            error="行情数据为空", lookback=lookback,
        )

    cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[cols].copy()
    df = df.sort_values("date").reset_index(drop=True)

    report_dt = pd.to_datetime(report_date)
    before = df[df["date"] <= report_dt].reset_index(drop=True)
    after = df[df["date"] > report_dt].reset_index(drop=True)

    if len(before) < lookback:
        return KronosPredResult(
            stock_code=stock_code, stock_name=stock_name,
            report_date=report_date, model_name=model_name,
            available=False, direction="", pred_pct_change=0,
            actual_pct_change=0, actual_direction="", hit=False,
            pred_days=0, pred_days_requested=pred_len,
            error=f"历史数据不足: {len(before)} < {lookback}", lookback=lookback,
        )

    if len(after) == 0:
        return KronosPredResult(
            stock_code=stock_code, stock_name=stock_name,
            report_date=report_date, model_name=model_name,
            available=False, direction="", pred_pct_change=0,
            actual_pct_change=0, actual_direction="", hit=False,
            pred_days=0, pred_days_requested=pred_len,
            error="报告日之后无交易日", lookback=lookback,
        )

    price_cols = ["open", "high", "low", "close", "volume"]
    x_df = before.iloc[-lookback:][price_cols].reset_index(drop=True)
    x_ts = before.iloc[-lookback:]["date"].reset_index(drop=True)

    actual_len = min(len(after), pred_len)
    y_ts = after.iloc[:actual_len]["date"].reset_index(drop=True)

    try:
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=actual_len, T=T, top_p=top_p,
            sample_count=sample_count, verbose=False,
        )
    except Exception as e:
        return KronosPredResult(
            stock_code=stock_code, stock_name=stock_name,
            report_date=report_date, model_name=model_name,
            available=False, direction="", pred_pct_change=0,
            actual_pct_change=0, actual_direction="", hit=False,
            pred_days=0, pred_days_requested=pred_len,
            error=f"预测失败: {e}", lookback=lookback,
        )

    actual_df = after.iloc[:actual_len].reset_index(drop=True)
    actual_close = actual_df["close"].values
    pred_close = pred_df["close"].values[:actual_len]

    a_total = (actual_close[-1] - actual_close[0]) / actual_close[0] * 100
    p_total = (pred_close[-1] - pred_close[0]) / pred_close[0] * 100

    # 预测方向
    if abs(p_total) < 1.0:
        direction = "震荡"
    elif p_total > 0:
        direction = "看多"
    else:
        direction = "看空"

    # 实际方向
    if abs(a_total) < 1.0:
        actual_direction = "震荡"
    elif a_total > 0:
        actual_direction = "看多"
    else:
        actual_direction = "看空"

    # 命中判定：方向一致算 hit
    hit = direction == actual_direction

    # 验证窗口不足的标记（实际数据太少，结果不可靠）
    min_verification_days = 5
    window_too_short = actual_len < min_verification_days

    return KronosPredResult(
        stock_code=stock_code, stock_name=stock_name,
        report_date=report_date, model_name=model_name,
        available=True, direction=direction,
        pred_pct_change=round(p_total, 2),
        actual_pct_change=round(a_total, 2),
        actual_direction=actual_direction,
        hit=hit, pred_days=actual_len,
        pred_days_requested=pred_len,
        error="", lookback=lookback,
        window_too_short=window_too_short,
        raw_pred_close=[round(float(v), 2) for v in pred_close],
        raw_actual_close=[round(float(v), 2) for v in actual_close],
    )


# ── 统计 ──────────────────────────────────────────────────────────────

@dataclass
class ModelStats:
    """单个模型的汇总统计。"""
    model_name: str
    total: int = 0
    available: int = 0
    hits: int = 0
    misses: int = 0
    no_direction: int = 0  # 预测为震荡的
    errors: int = 0

    # 按方向细分
    bullish: int = 0
    bullish_hits: int = 0
    bearish: int = 0
    bearish_hits: int = 0

    # 按震荡幅度判定
    shake_hits: int = 0
    shake_misses: int = 0

    @property
    def hit_rate(self) -> float:
        judged = self.hits + self.misses
        return (self.hits / judged * 100) if judged else 0.0

    @property
    def judged(self) -> int:
        return self.hits + self.misses


def compute_stats(results: List[KronosPredResult], model_name: str) -> ModelStats:
    stats = ModelStats(model_name=model_name)
    for r in results:
        if r.model_name != model_name:
            continue
        stats.total += 1
        if not r.available:
            stats.errors += 1
            continue
        # 跳过验证窗口不足的预测（不纳入命中率计算）
        if r.window_too_short:
            continue
        stats.available += 1
        if r.direction == "震荡":
            stats.no_direction += 1
            if r.actual_direction == "震荡":
                stats.shake_hits += 1
                stats.hits += 1
            else:
                stats.shake_misses += 1
                stats.misses += 1
        elif r.direction == "看多":
            stats.bullish += 1
            if r.hit:
                stats.bullish_hits += 1
                stats.hits += 1
            else:
                stats.misses += 1
        elif r.direction == "看空":
            stats.bearish += 1
            if r.hit:
                stats.bearish_hits += 1
                stats.hits += 1
            else:
                stats.misses += 1
    return stats


# ── 报告生成 ──────────────────────────────────────────────────────────

def generate_report(
    base_results: List[KronosPredResult],
    small_results: List[KronosPredResult],
    dates: List[str],
) -> str:
    base_stats = compute_stats(base_results, "Kronos-base")
    small_stats = compute_stats(small_results, "Kronos-small")

    lines = []
    lines.append("# Kronos-base vs Kronos-small 批量回测对比报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"预测基准日: {dates[0]} ~ {dates[-1]}")
    lines.append(f"追踪股票: {len(TRACKED_STOCKS)} 只")
    lines.append(f"预测天数: {PRED_LEN} 个交易日\n")
    lines.append(f"Kronos-base: 102.3M params | Kronos-small: 24.7M params\n")

    # 统计窗口不足的样本
    too_short_base = sum(1 for r in base_results if r.available and r.window_too_short)
    too_short_small = sum(1 for r in small_results if r.available and r.window_too_short)
    too_short_dates = sorted(set(
        r.report_date for r in base_results if r.available and r.window_too_short
    ))
    if too_short_dates:
        lines.append(f"> ⚠️ 验证窗口不足（实际交易日 < 5）已排除: {too_short_base}/{len(base_results)} base, {too_short_small}/{len(small_results)} small")
        lines.append(f"> 排除日期: {', '.join(too_short_dates)}")
        lines.append(f"> 原因: 这些日期的 15 日预测窗口尚未走完足够交易日，结果不可靠。\n")

    # 总览对比
    lines.append("## 总览对比\n")
    lines.append("| 指标 | Kronos-base (102M) | Kronos-small (25M) | 差异 |")
    lines.append("|------|-------------------|--------------------|------|")
    b_rate = base_stats.hit_rate
    s_rate = small_stats.hit_rate
    diff = b_rate - s_rate
    diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
    emoji = "🟢" if diff > 0 else ("🔴" if diff < 0 else "⚪")
    lines.append(f"| 总预测数 | {base_stats.total} | {small_stats.total} | - |")
    lines.append(f"| 有效预测 | {base_stats.available} | {small_stats.available} | - |")
    lines.append(f"| 总命中率 | **{b_rate:.1f}%** ({base_stats.hits}/{base_stats.judged}) | **{s_rate:.1f}%** ({small_stats.hits}/{small_stats.judged}) | {emoji} {diff_str} |")
    lines.append(f"| 预测震荡 | {base_stats.no_direction} | {small_stats.no_direction} | - |")
    lines.append(f"| 预测看多 | {base_stats.bullish} | {small_stats.bullish} | - |")
    lines.append(f"| 预测看空 | {base_stats.bearish} | {small_stats.bearish} | - |")
    lines.append(f"| 错误数 | {base_stats.errors} | {small_stats.errors} | - |")

    # 方向细分
    lines.append("\n## 方向细分\n")
    b_bull_rate = (base_stats.bullish_hits / base_stats.bullish * 100) if base_stats.bullish else 0
    s_bull_rate = (small_stats.bullish_hits / small_stats.bullish * 100) if small_stats.bullish else 0
    b_bear_rate = (base_stats.bearish_hits / base_stats.bearish * 100) if base_stats.bearish else 0
    s_bear_rate = (small_stats.bearish_hits / small_stats.bearish * 100) if small_stats.bearish else 0

    lines.append("| 方向 | Kronos-base | Kronos-small | 差异 |")
    lines.append("|------|-------------|-------------|------|")
    bull_diff = b_bull_rate - s_bull_rate
    bear_diff = b_bear_rate - s_bear_rate
    lines.append(f"| 看多 | {b_bull_rate:.1f}% ({base_stats.bullish_hits}/{base_stats.bullish}) | {s_bull_rate:.1f}% ({small_stats.bullish_hits}/{small_stats.bullish}) | {'🟢 +' if bull_diff > 0 else ('🔴 ' if bull_diff < 0 else '⚪ ')}{bull_diff:+.1f}% |")
    lines.append(f"| 看空 | {b_bear_rate:.1f}% ({base_stats.bearish_hits}/{base_stats.bearish}) | {s_bear_rate:.1f}% ({small_stats.bearish_hits}/{small_stats.bearish}) | {'🟢 +' if bear_diff > 0 else ('🔴 ' if bear_diff < 0 else '⚪ ')}{bear_diff:+.1f}% |")

    # 按日期
    lines.append("\n## 按日期\n")
    lines.append("| 日期 | Kronos-base hit | Kronos-base 有效 | Kronos-base 命中率 | Kronos-small hit | Kronos-small 有效 | Kronos-small 命中率 |")
    lines.append("|------|----------------|-----------------|-------------------|-----------------|------------------|--------------------|")
    for d in dates:
        b_h = sum(1 for r in base_results if r.report_date == d and r.available and not r.window_too_short and r.hit)
        b_j = sum(1 for r in base_results if r.report_date == d and r.available and not r.window_too_short)
        s_h = sum(1 for r in small_results if r.report_date == d and r.available and not r.window_too_short and r.hit)
        s_j = sum(1 for r in small_results if r.report_date == d and r.available and not r.window_too_short)
        b_r = f"{b_h/b_j*100:.1f}%" if b_j else "-"
        s_r = f"{s_h/s_j*100:.1f}%" if s_j else "-"
        lines.append(f"| {d} | {b_h} | {b_j} | {b_r} | {s_h} | {s_j} | {s_r} |")

    # 按股票
    lines.append("\n## 按股票\n")
    lines.append("| 股票 | Kronos-base hit | Kronos-base 有效 | Kronos-base 命中率 | Kronos-small hit | Kronos-small 有效 | Kronos-small 命中率 |")
    lines.append("|------|----------------|-----------------|-------------------|-----------------|------------------|--------------------|")
    for code, name in TRACKED_STOCKS:
        b_h = sum(1 for r in base_results if r.stock_code == code and r.available and not r.window_too_short and r.hit)
        b_j = sum(1 for r in base_results if r.stock_code == code and r.available and not r.window_too_short)
        s_h = sum(1 for r in small_results if r.stock_code == code and r.available and not r.window_too_short and r.hit)
        s_j = sum(1 for r in small_results if r.stock_code == code and r.available and not r.window_too_short)
        b_r = f"{b_h/b_j*100:.1f}%" if b_j else "-"
        s_r = f"{s_h/s_j*100:.1f}%" if s_j else "-"
        label = f"{code} {name}"
        lines.append(f"| {label} | {b_h} | {b_j} | {b_r} | {s_h} | {s_j} | {s_r} |")

    # 详细逐条结果
    lines.append("\n## 逐条详细结果\n")
    lines.append("| 日期 | 股票 | base 方向 | base 预测涨跌幅 | 实际方向 | 实际涨跌幅 | base 命中 | small 方向 | small 预测涨跌幅 | small 命中 |")
    lines.append("|------|------|----------|----------------|----------|-----------|-----------|-----------|-----------------|-----------|")
    for d in dates:
        for code, name in TRACKED_STOCKS:
            br = [r for r in base_results if r.report_date == d and r.stock_code == code]
            sr = [r for r in small_results if r.report_date == d and r.stock_code == code]
            b = br[0] if br else None
            s = sr[0] if sr else None
            label = f"{code} {name}"
            if b and b.available:
                b_line = f"{b.direction} | {b.pred_pct_change:+.2f}% | {b.actual_direction} | {b.actual_pct_change:+.2f}% | {'✅' if b.hit else '❌'} |"
            elif b:
                b_line = f"❌ {b.error} | ❌ | ❌ | ❌ | ❌ |"
            else:
                b_line = "- | - | - | - | - |"

            if s and s.available:
                s_line = f"{s.direction} | {s.pred_pct_change:+.2f}% | {'✅' if s.hit else '❌'} |"
            elif s:
                s_line = f"❌ {s.error} | ❌ |"
            else:
                s_line = "- | - |"

            lines.append(f"| {d} | {label} | {b_line} {s_line}")

    # 结论
    lines.append("\n## 结论\n")
    if diff > 0:
        lines.append(f"🟢 **Kronos-base (102M) 整体命中率高于 Kronos-small (25M) {diff:.1f}pp**")
    elif diff < 0:
        lines.append(f"🔴 **Kronos-base (102M) 整体命中率低于 Kronos-small (25M) {-diff:.1f}pp**")
    else:
        lines.append("⚪ **两者整体命中率持平**")

    lines.append(f"\n- 有效预测样本数: {base_stats.available}（base）/ {small_stats.available}（small）")
    lines.append(f"- 预测区间: 全量历史报告 {PREDICTION_DATES_FULL[0]} ~ {PREDICTION_DATES_FULL[-1]} + 上周 {PREDICTION_DATES_LAST_WEEK[0]} ~ {PREDICTION_DATES_LAST_WEEK[-1]}")
    lines.append(f"  - 全量报告: {len(PREDICTION_DATES_FULL)} 个交易日，15 日窗口已完全走完")
    lines.append(f"  - 上周: {len(PREDICTION_DATES_LAST_WEEK)} 个交易日，仅部分窗口可验证")
    lines.append(f"- 验证窗口不足（<5 交易日）已排除: {too_short_base} base / {too_short_small} small")
    lines.append(f"- 由于上周的 15 日预测窗口尚未完全走完（今天 2026-08-04），")
    lines.append(f"  上周的实际涨跌幅只能反映截至当前的数据。完整回测建议等 8 月中旬窗口完全关闭后重跑。")

    return "\n".join(lines) + "\n"


# ── 主函数 ────────────────────────────────────────────────────────────

def run_batch(dates: List[str], label: str, fetcher_mgr: DataFetcherManager) -> Tuple[str, List[KronosPredResult], List[KronosPredResult]]:
    """对一组日期运行双模型预测，返回 (label, base_results, small_results)。"""
    all_results: List[KronosPredResult] = []
    total_tasks = len(dates) * len(TRACKED_STOCKS) * 2
    logger.info("开始回测 [%s]: %d 个交易日 × %d 只股票 × 2 模型 = %d 次预测",
                label, len(dates), len(TRACKED_STOCKS), total_tasks)

    def run_task(stock_code: str, stock_name: str, date: str, model: str) -> KronosPredResult:
        return run_prediction(stock_code, stock_name, date, model, fetcher_mgr)

    task_id = 0
    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for date in dates:
            for code, name in TRACKED_STOCKS:
                for model in ["Kronos-base", "Kronos-small"]:
                    task_id += 1
                    future = pool.submit(run_task, code, name, date, model)
                    futures[future] = (date, code, name, model)

        for future in as_completed(futures):
            date, code, name, model = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                status = "✅" if result.available else "❌"
                logger.info("  [%s-%s] %s %s %s: %s %s %+.2f%%",
                            label, status, date, code, name, model,
                            result.direction if result.available else result.error,
                            result.pred_pct_change if result.available else 0)
            except Exception as e:
                logger.error("  [%s-❌] %s %s %s %s: %s", label, date, code, name, model, e)

    base_results = [r for r in all_results if r.model_name == "Kronos-base"]
    small_results = [r for r in all_results if r.model_name == "Kronos-small"]
    return label, base_results, small_results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    out_dir = ROOT / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    fetcher_mgr = DataFetcherManager()

    # 运行两组回测
    full_label, full_base, full_small = run_batch(
        PREDICTION_DATES_FULL, "full_history", fetcher_mgr)
    week_label, week_base, week_small = run_batch(
        PREDICTION_DATES_LAST_WEEK, "last_week", fetcher_mgr)

    # 合并两组数据
    all_base = full_base + week_base
    all_small = full_small + week_small
    all_dates = PREDICTION_DATES_FULL + PREDICTION_DATES_LAST_WEEK

    # 生成报告（合并）
    report = generate_report(all_base, all_small, all_dates)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"kronos_base_compare_{ts}.md"
    report_path.write_text(report, encoding="utf-8")

    # 同时保存 JSONL 原始数据
    jsonl_path = out_dir / f"kronos_base_compare_{ts}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in all_base + all_small:
            f.write(json.dumps(asdict(r), ensure_ascii=False, default=str) + "\n")

    logger.info("\n" + report)
    logger.info(f"报告已保存: {report_path}")
    logger.info(f"原始数据: {jsonl_path}")


if __name__ == "__main__":
    main()