#!/usr/bin/env python3
"""Kronos 参数搜索：在代表性子集上测试不同参数组合的效果。

目的：
    1. 找到最优参数组合，系统性降低 MAPE 和偏误
    2. 测试 T、top_p、sample_count、lookback 的影响

用法:
    cd daily_stock_analysis
    python -m evals.kronos_sweep
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# 项目路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Kronos 路径
KRONOS_PATH = "/tmp/Kronos"
if KRONOS_PATH not in sys.path:
    sys.path.insert(0, KRONOS_PATH)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

logger = logging.getLogger("kronos_sweep")

# =========================================================================
# 测试集：3 只代表股票 × 2 个报告日 = 6 个任务
# =========================================================================
SWEEP_STOCKS = [
    ("300308", "中际旭创"),  # 高价、高 MAPE、趋势方向好
    ("002463", "沪电股份"),  # 表现全面好
    ("300031", "宝通科技"),  # 低价、趋势方向差
    ("300502", "新易盛"),    # 高价、高 MAPE、趋势方向好
    ("603019", "中科曙光"),  # 低价、趋势方向差
]
SWEEP_DATES = ["2026-07-02", "2026-07-17"]

# =========================================================================
# 参数搜索空间
# =========================================================================
# Phase 1: T × top_p（固定 lookback=400, sample_count=1）
T_VALUES = [0.5, 0.8, 1.0, 1.2, 1.5]
TOP_P_VALUES = [0.85, 0.9, 0.95]

# Phase 2: sample_count（固定 best T, best top_p, lookback=400）
SAMPLE_COUNT_VALUES = [1, 3, 5, 10, 20]

# Phase 3: lookback（固定 best T, best top_p, best sample_count）
LOOKBACK_VALUES = [200, 400, 600, 800]

# =========================================================================
# 模型加载（复用 kronos_eval 的方式）
# =========================================================================
_kronos_predictor = None


def _load_kronos():
    global _kronos_predictor
    if _kronos_predictor is not None:
        return _kronos_predictor
    import torch
    from model import Kronos, KronosTokenizer, KronosPredictor
    hf_name = "NeoQuasar/Kronos-small"
    hf_tokenizer = "NeoQuasar/Kronos-Tokenizer-base"
    tokenizer = KronosTokenizer.from_pretrained(hf_tokenizer, token=False)
    model = Kronos.from_pretrained(hf_name, token=False)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Kronos-small 参数: %.1fM", params)
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    _kronos_predictor = predictor
    return predictor


# =========================================================================
# 数据获取
# =========================================================================
def _fetch_data(fetcher, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    from data_provider.tencent_fetcher import TencentFetcher
    df = fetcher.get_daily_data(symbol, start_date=start_date, end_date=end_date)
    if df.empty:
        return df
    cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[cols].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


# =========================================================================
# 单次预测
# =========================================================================
def run_prediction(predictor, df, report_date, lookback, pred_len, T, top_p, sample_count):
    report_dt = pd.to_datetime(report_date)
    before = df[df["date"] <= report_dt].reset_index(drop=True)
    after = df[df["date"] > report_dt].reset_index(drop=True)
    if len(before) < lookback or len(after) == 0:
        return None, None
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    x_df = before.iloc[-lookback:][price_cols].reset_index(drop=True)
    x_ts = before.iloc[-lookback:]['date'].reset_index(drop=True)
    actual_len = min(len(after), pred_len)
    y_ts = after.iloc[:actual_len]['date'].reset_index(drop=True)
    try:
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=actual_len, T=T, top_p=top_p,
            sample_count=sample_count, verbose=False,
        )
    except Exception as e:
        return None, None
    actual_df = after.iloc[:actual_len].reset_index(drop=True)
    return pred_df, actual_df


# =========================================================================
# 指标计算
# =========================================================================
def calc_metrics(actual_close, pred_close):
    n = min(len(actual_close), len(pred_close))
    a, p = actual_close[:n], pred_close[:n]
    if n == 0:
        return {}
    mape = float(np.mean(np.abs((a - p) / a)) * 100)
    if n > 1:
        a_ret, p_ret = np.diff(a), np.diff(p)
        dir_acc = float(np.mean(np.sign(a_ret) == np.sign(p_ret)) * 100)
    else:
        dir_acc = 0.0
    a_total = float((a[-1] - a[0]) / a[0] * 100)
    p_total = float((p[-1] - p[0]) / p[0] * 100)
    total_dir_correct = bool(np.sign(a_total) == np.sign(p_total))
    mean_bias = float(np.mean(p - a))
    # RMSE
    rmse = float(np.sqrt(np.mean((p - a) ** 2)))
    return {
        "mape": round(mape, 2),
        "rmse": round(rmse, 2),
        "direction_accuracy": round(dir_acc, 1),
        "actual_total_return_pct": round(a_total, 2),
        "pred_total_return_pct": round(p_total, 2),
        "total_direction_correct": total_dir_correct,
        "mean_bias": round(mean_bias, 2),
        "pred_days": n,
    }


# =========================================================================
# 数据缓存：避免重复拉取
# =========================================================================
_data_cache = {}


def get_cached_data(fetcher, code, lookback, pred_len, report_date):
    """获取并缓存行情数据（按 code 缓存，与参数无关）。"""
    if code in _data_cache:
        return _data_cache[code]
    # 请求 800 天前的数据（覆盖 lookback=800 的场景），但不超过 2023-01-01
    earliest = pd.to_datetime("2023-01-01")
    start_dt = max(pd.to_datetime(report_date) - timedelta(days=800), earliest)
    end_dt = pd.to_datetime(report_date) + timedelta(days=pred_len * 2)
    df = _fetch_data(fetcher, code, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    if not df.empty:
        _data_cache[code] = df
    return df


# =========================================================================
# 主流程
# =========================================================================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    out_dir = Path("evals/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_len = 15

    # 加载模型
    print("🤗 加载 Kronos 模型...", file=sys.stderr)
    predictor = _load_kronos()

    # 初始化数据源
    from data_provider.tencent_fetcher import TencentFetcher
    fetcher = TencentFetcher()

    # 预取数据
    print("📥 预取行情数据...", file=sys.stderr)
    for code, name in SWEEP_STOCKS:
        for d in SWEEP_DATES:
            get_cached_data(fetcher, code, 800, pred_len, d)
    print(f"   已缓存 {len(_data_cache)} 只股票", file=sys.stderr)

    # =====================================================================
    # Phase 1: T × top_p sweep
    # =====================================================================
    print("\n" + "=" * 70, file=sys.stderr)
    print("Phase 1: T × top_p 搜索 (lookback=400, sample_count=1)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    phase1_results = []
    total = len(T_VALUES) * len(TOP_P_VALUES) * len(SWEEP_STOCKS) * len(SWEEP_DATES)
    count = 0
    t0 = time.time()

    for T in T_VALUES:
        for top_p in TOP_P_VALUES:
            params = {"lookback": 400, "T": T, "top_p": top_p, "sample_count": 1}
            for code, name in SWEEP_STOCKS:
                for report_date in SWEEP_DATES:
                    count += 1
                    if count % 10 == 0:
                        el = time.time() - t0
                        print(f"  [{count}/{total}] T={T} top_p={top_p} {name}({code}) {report_date} - {el:.0f}s", file=sys.stderr)

                    df = _data_cache.get(code)
                    if df is None:
                        continue

                    pred_df, actual_df = run_prediction(
                        predictor, df, report_date, params["lookback"], pred_len,
                        T, top_p, params["sample_count"],
                    )
                    if pred_df is None:
                        continue

                    metrics = calc_metrics(actual_df["close"].values, pred_df["close"].values)
                    if not metrics:
                        continue

                    result = {
                        "report_date": report_date,
                        "code": code,
                        "name": name,
                        **params,
                        **metrics,
                    }
                    phase1_results.append(result)

    elapsed = time.time() - t0
    print(f"  Phase 1 完成: {len(phase1_results)} 条结果, {elapsed:.1f}s", file=sys.stderr)

    if not phase1_results:
        print("❌ Phase 1 无有效结果!", file=sys.stderr)
        sys.exit(1)

    # Phase 1 分析：按参数组合聚合
    phase1_summary = defaultdict(list)
    for r in phase1_results:
        key = (r["T"], r["top_p"])
        phase1_summary[key].append(r)

    print("\nPhase 1: T × top_p 汇总", file=sys.stderr)
    print(f"{'T':>5} {'top_p':>6} {'n':>4} {'MAPE':>7} {'RMSE':>8} {'DirAcc':>7} {'Trend':>7} {'Bias':>8}", file=sys.stderr)
    print("-" * 55, file=sys.stderr)
    best_t_tp = None
    best_mape = float("inf")
    for (T, top_p), rs in sorted(phase1_summary.items()):
        avg_mape = np.mean([r["mape"] for r in rs])
        avg_rmse = np.mean([r["rmse"] for r in rs])
        avg_dir = np.mean([r["direction_accuracy"] for r in rs])
        trend_correct = sum(1 for r in rs if r["total_direction_correct"])
        avg_bias = np.mean([r["mean_bias"] for r in rs])
        print(f"{T:>5.1f} {top_p:>6.2f} {len(rs):>4d} {avg_mape:>7.2f} {avg_rmse:>8.1f} {avg_dir:>7.1f} {trend_correct}/{len(rs):>2d} {avg_bias:>+8.2f}", file=sys.stderr)
        if avg_mape < best_mape:
            best_mape = avg_mape
            best_t_tp = (T, top_p)

    best_T, best_top_p = best_t_tp
    print(f"\n🏆 Phase 1 最佳: T={best_T}, top_p={best_top_p} (MAPE={best_mape:.2f}%)", file=sys.stderr)

    # =====================================================================
    # Phase 2: sample_count sweep
    # =====================================================================
    print("\n" + "=" * 70, file=sys.stderr)
    print(f"Phase 2: sample_count 搜索 (T={best_T}, top_p={best_top_p}, lookback=400)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    phase2_results = []
    total2 = len(SAMPLE_COUNT_VALUES) * len(SWEEP_STOCKS) * len(SWEEP_DATES)
    count2 = 0
    t0 = time.time()

    for sc in SAMPLE_COUNT_VALUES:
        for code, name in SWEEP_STOCKS:
            for report_date in SWEEP_DATES:
                count2 += 1
                if count2 % 10 == 0:
                    el = time.time() - t0
                    print(f"  [{count2}/{total2}] sample_count={sc} {name}({code}) {report_date} - {el:.0f}s", file=sys.stderr)

                df = _data_cache.get(code)
                if df is None:
                    continue

                pred_df, actual_df = run_prediction(
                    predictor, df, report_date, 400, pred_len,
                    best_T, best_top_p, sc,
                )
                if pred_df is None:
                    continue

                metrics = calc_metrics(actual_df["close"].values, pred_df["close"].values)
                if not metrics:
                    continue

                result = {
                    "report_date": report_date,
                    "code": code,
                    "name": name,
                    "lookback": 400, "T": best_T, "top_p": best_top_p,
                    "sample_count": sc,
                    **metrics,
                }
                phase2_results.append(result)

    elapsed = time.time() - t0
    print(f"  Phase 2 完成: {len(phase2_results)} 条结果, {elapsed:.1f}s", file=sys.stderr)

    phase2_summary = defaultdict(list)
    for r in phase2_results:
        phase2_summary[r["sample_count"]].append(r)

    print(f"\nPhase 2: sample_count 汇总 (T={best_T}, top_p={best_top_p})", file=sys.stderr)
    print(f"{'sc':>5} {'n':>4} {'MAPE':>7} {'RMSE':>8} {'DirAcc':>7} {'Trend':>7} {'Bias':>8}", file=sys.stderr)
    print("-" * 55, file=sys.stderr)
    best_sc = None
    best_mape2 = float("inf")
    for sc in sorted(phase2_summary.keys()):
        rs = phase2_summary[sc]
        avg_mape = np.mean([r["mape"] for r in rs])
        avg_rmse = np.mean([r["rmse"] for r in rs])
        avg_dir = np.mean([r["direction_accuracy"] for r in rs])
        trend_correct = sum(1 for r in rs if r["total_direction_correct"])
        avg_bias = np.mean([r["mean_bias"] for r in rs])
        print(f"{sc:>5d} {len(rs):>4d} {avg_mape:>7.2f} {avg_rmse:>8.1f} {avg_dir:>7.1f} {trend_correct}/{len(rs):>2d} {avg_bias:>+8.2f}", file=sys.stderr)
        if avg_mape < best_mape2:
            best_mape2 = avg_mape
            best_sc = sc

    print(f"\n🏆 Phase 2 最佳: sample_count={best_sc} (MAPE={best_mape2:.2f}%)", file=sys.stderr)

    # =====================================================================
    # Phase 3: lookback sweep
    # =====================================================================
    print("\n" + "=" * 70, file=sys.stderr)
    print(f"Phase 3: lookback 搜索 (T={best_T}, top_p={best_top_p}, sample_count={best_sc})", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    phase3_results = []
    total3 = len(LOOKBACK_VALUES) * len(SWEEP_STOCKS) * len(SWEEP_DATES)
    count3 = 0
    t0 = time.time()

    for lb in LOOKBACK_VALUES:
        for code, name in SWEEP_STOCKS:
            for report_date in SWEEP_DATES:
                count3 += 1
                if count3 % 10 == 0:
                    el = time.time() - t0
                    print(f"  [{count3}/{total3}] lookback={lb} {name}({code}) {report_date} - {el:.0f}s", file=sys.stderr)

                df = _data_cache.get(code)
                if df is None:
                    continue

                pred_df, actual_df = run_prediction(
                    predictor, df, report_date, lb, pred_len,
                    best_T, best_top_p, best_sc,
                )
                if pred_df is None:
                    continue

                metrics = calc_metrics(actual_df["close"].values, pred_df["close"].values)
                if not metrics:
                    continue

                result = {
                    "report_date": report_date,
                    "code": code,
                    "name": name,
                    "lookback": lb, "T": best_T, "top_p": best_top_p,
                    "sample_count": best_sc,
                    **metrics,
                }
                phase3_results.append(result)

    elapsed = time.time() - t0
    print(f"  Phase 3 完成: {len(phase3_results)} 条结果, {elapsed:.1f}s", file=sys.stderr)

    phase3_summary = defaultdict(list)
    for r in phase3_results:
        phase3_summary[r["lookback"]].append(r)

    print(f"\nPhase 3: lookback 汇总 (T={best_T}, top_p={best_top_p}, sample_count={best_sc})", file=sys.stderr)
    print(f"{'lb':>5} {'n':>4} {'MAPE':>7} {'RMSE':>8} {'DirAcc':>7} {'Trend':>7} {'Bias':>8}", file=sys.stderr)
    print("-" * 55, file=sys.stderr)
    best_lb = None
    best_mape3 = float("inf")
    for lb in sorted(phase3_summary.keys()):
        rs = phase3_summary[lb]
        avg_mape = np.mean([r["mape"] for r in rs])
        avg_rmse = np.mean([r["rmse"] for r in rs])
        avg_dir = np.mean([r["direction_accuracy"] for r in rs])
        trend_correct = sum(1 for r in rs if r["total_direction_correct"])
        avg_bias = np.mean([r["mean_bias"] for r in rs])
        print(f"{lb:>5d} {len(rs):>4d} {avg_mape:>7.2f} {avg_rmse:>8.1f} {avg_dir:>7.1f} {trend_correct}/{len(rs):>2d} {avg_bias:>+8.2f}", file=sys.stderr)
        if avg_mape < best_mape3:
            best_mape3 = avg_mape
            best_lb = lb

    print(f"\n🏆 Phase 3 最佳: lookback={best_lb} (MAPE={best_mape3:.2f}%)", file=sys.stderr)

    # =====================================================================
    # 最终推荐
    # =====================================================================
    print("\n" + "=" * 70, file=sys.stderr)
    print("🏆 最终推荐参数组合", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"  T           = {best_T}", file=sys.stderr)
    print(f"  top_p       = {best_top_p}", file=sys.stderr)
    print(f"  sample_count = {best_sc}", file=sys.stderr)
    print(f"  lookback    = {best_lb}", file=sys.stderr)
    print(f"  MAPE        = {best_mape3:.2f}% (vs baseline 21.98%)", file=sys.stderr)
    print(file=sys.stderr)

    # 保存结果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"kronos_sweep_{ts}.json"
    all_results = {
        "phase1": phase1_results,
        "phase2": phase2_results,
        "phase3": phase3_results,
        "best_params": {
            "T": best_T,
            "top_p": best_top_p,
            "sample_count": best_sc,
            "lookback": best_lb,
        },
        "baseline_mape": 21.98,
        "best_mape": best_mape3,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  💾 结果已保存: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()