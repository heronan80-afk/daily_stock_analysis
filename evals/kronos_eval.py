#!/usr/bin/env python3
"""Kronos 回测 CLI：对历史报告中的股票批量跑 Kronos 预测并评估准确率。

用法:
    cd daily_stock_analysis
    python -m evals.kronos_eval --reports reports/ --out evals/results/
    python -m evals.kronos_eval --reports reports/ --model Kronos-small --lookback 400 --pred-len 15
    python -m evals.kronos_eval --reports reports/ --report-date 2026-07-10  # 单日调试

输出:
    evals/results/kronos_outcomes_<ts>.jsonl   逐条预测结果
    evals/results/kronos_summary_<ts>.md       汇总统计
"""
from __future__ import annotations

import argparse
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

# 项目路径支持
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evals.extract_signals import load_all_signals, parse_report  # noqa: E402
    from evals.kronos_params import get_stock_params  # noqa: E402
else:
    from .extract_signals import load_all_signals, parse_report
    from .kronos_params import get_stock_params

# Kronos 路径
KRONOS_PATH = os.environ.get("KRONOS_PATH", os.path.expanduser("~/kronos/Kronos"))
if KRONOS_PATH not in sys.path:
    sys.path.insert(0, KRONOS_PATH)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

logger = logging.getLogger("kronos_eval")

# ---------------------------------------------------------------------------
# 模型加载（懒加载单例）
# ---------------------------------------------------------------------------
_kronos_predictor = None
_kronos_model_name = None
_kronos_device = None


def _load_kronos(model_name: str = "Kronos-small", device: str = "cpu"):
    """加载 Kronos 模型（缓存）。"""
    global _kronos_predictor, _kronos_model_name, _kronos_device
    if _kronos_predictor is not None and _kronos_model_name == model_name and _kronos_device == device:
        return _kronos_predictor

    import torch
    from model import Kronos, KronosTokenizer, KronosPredictor

    hf_name = f"NeoQuasar/{model_name}"
    hf_tokenizer = f"NeoQuasar/Kronos-Tokenizer-base"

    logger.info("🤗 加载 Kronos 模型: %s (device=%s)", hf_name, device)
    t0 = time.time()

    tokenizer = KronosTokenizer.from_pretrained(hf_tokenizer, token=False)
    model = Kronos.from_pretrained(hf_name, token=False)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("   参数: %.1fM, 加载耗时: %.1fs", params, time.time() - t0)

    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    _kronos_predictor = predictor
    _kronos_model_name = model_name
    _kronos_device = device
    return predictor


# ---------------------------------------------------------------------------
# 数据获取
# ---------------------------------------------------------------------------
def _fetch_data(fetcher, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取个股日线数据，返回按 date 排序的 df。"""
    from data_provider.tencent_fetcher import TencentFetcher
    df = fetcher.get_daily_data(symbol, start_date=start_date, end_date=end_date)
    if df.empty:
        return df
    cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[cols].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Kronos 预测
# ---------------------------------------------------------------------------
def run_kronos_prediction(
    predictor, df: pd.DataFrame, report_date: str,
    lookback: int = 400, pred_len: int = 15,
    T: float = 1.0, top_p: float = 0.9,
    sample_count: int = 1,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """对一只股票跑 Kronos 预测。

    Returns:
        (pred_df, actual_df, lookback_df, error_msg)
        pred_df/actual_df 为 None 表示预测失败，error_msg 说明原因。
    """
    report_dt = pd.to_datetime(report_date)
    before = df[df["date"] <= report_dt].reset_index(drop=True)
    after = df[df["date"] > report_dt].reset_index(drop=True)

    if len(before) < lookback:
        return None, None, None, f"历史数据不足: {len(before)} < {lookback}"

    if len(after) == 0:
        return None, None, None, "报告日之后无交易日"

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
        return None, None, None, f"预测失败: {e}"

    actual_df = after.iloc[:actual_len].reset_index(drop=True)
    return pred_df, actual_df, before, None


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def _to_native(v):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _native_dict(d: dict) -> dict:
    return {k: _to_native(v) for k, v in d.items()}


def calc_metrics(actual_close: np.ndarray, pred_close: np.ndarray) -> dict:
    """计算预测准确率指标。"""
    n = min(len(actual_close), len(pred_close))
    a, p = actual_close[:n], pred_close[:n]

    if n == 0:
        return {"error": "无数据"}

    # MAPE
    mape = float(np.mean(np.abs((a - p) / a)) * 100)

    # 日均方向准确率
    if n > 1:
        a_ret, p_ret = np.diff(a), np.diff(p)
        dir_acc = float(np.mean(np.sign(a_ret) == np.sign(p_ret)) * 100)
    else:
        dir_acc = 0.0

    # 总涨跌对比
    a_total = float((a[-1] - a[0]) / a[0] * 100)
    p_total = float((p[-1] - p[0]) / p[0] * 100)
    total_dir_correct = bool(np.sign(a_total) == np.sign(p_total))

    # 最大回撤对比
    a_peak = a[0]
    a_max_dd = 0.0
    for v in a:
        if v > a_peak:
            a_peak = v
        dd = (a_peak - v) / a_peak * 100
        if dd > a_max_dd:
            a_max_dd = dd

    p_peak = p[0]
    p_max_dd = 0.0
    for v in p:
        if v > p_peak:
            p_peak = v
        dd = (p_peak - v) / p_peak * 100
        if dd > p_max_dd:
            p_max_dd = dd

    # 价格偏差（绝对值，反映系统性偏大/偏小）
    mean_bias = float(np.mean(p - a))

    return _native_dict({
        "mape": round(mape, 2),
        "direction_accuracy": round(dir_acc, 1),
        "actual_total_return_pct": round(a_total, 2),
        "pred_total_return_pct": round(p_total, 2),
        "total_direction_correct": total_dir_correct,
        "actual_max_drawdown_pct": round(a_max_dd, 2),
        "pred_max_drawdown_pct": round(p_max_dd, 2),
        "mean_bias": round(mean_bias, 2),
        "pred_days": n,
    })


# ---------------------------------------------------------------------------
# 核心：单股票预测 + 评估
# ---------------------------------------------------------------------------
def evaluate_one(
    predictor, fetcher, code: str, name: str, report_date: str,
    lookback: int, pred_len: int, T: float, top_p: float,
    use_per_stock: bool = False,
) -> dict:
    """对一只股票在给定报告日跑预测并评估。

    use_per_stock=True 时，lookback/T/top_p 被忽略，改用 kronos_params 中的个股配置。

    Returns:
        dict 结果，包含 error 字段表示失败原因。
    """
    result = {
        "report_date": report_date,
        "code": code,
        "name": name,
    }

    # 个股参数模式：覆盖 CLI 参数
    sample_count = 1
    if use_per_stock:
        lookback, T, top_p, sample_count = get_stock_params(code)
        result["_params"] = f"per_stock:lb={lookback},T={T},top_p={top_p},sc={sample_count}"
    else:
        result["_params"] = f"cli:lb={lookback},T={T},top_p={top_p},sc=1"

    # 数据范围：lookback 天之前 + pred_len 天之后 + 缓冲
    start_dt = pd.to_datetime(report_date) - timedelta(days=lookback * 2)  # 留余量
    end_dt = pd.to_datetime(report_date) + timedelta(days=pred_len * 2)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    df = _fetch_data(fetcher, code, start_str, end_str)
    if df.empty:
        result["error"] = "取行情为空"
        return result

    result["data_days"] = len(df)
    result["data_start"] = str(df["date"].min())
    result["data_end"] = str(df["date"].max())

    pred_df, actual_df, before_df, error = run_kronos_prediction(
        predictor, df, report_date, lookback, pred_len, T, top_p, sample_count,
    )
    if error:
        result["error"] = error
        return result

    # 报告日收盘价
    last_before = before_df.iloc[-1]
    result["last_price"] = round(float(last_before["close"]), 2)

    # 指标
    metrics = calc_metrics(
        actual_df["close"].values,
        pred_df["close"].values[:len(actual_df)],
    )
    result.update(metrics)

    # 详细预测序列（可选，为节省空间不在 JSONL 主记录中保留）
    result["_pred_series"] = {
        "dates": [str(d) for d in actual_df["date"].values],
        "actual_close": [round(float(v), 2) for v in actual_df["close"].values],
        "pred_close": [round(float(v), 2) for v in pred_df["close"].values[:len(actual_df)]],
    }

    return result


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def _fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def summarize(results: List[dict]) -> str:
    """生成 Markdown 汇总报告。"""
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    lines = []
    lines.append("# Kronos 回测报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"总样本: {len(results)} | 成功: {len(valid)} | 失败: {len(errors)}\n")

    if not valid:
        lines.append("\n## ❌ 无有效结果\n")
        return "\n".join(lines)

    # 整体统计
    all_mape = [r["mape"] for r in valid]
    all_dir = [r["direction_accuracy"] for r in valid]
    all_trend = [r["total_direction_correct"] for r in valid]
    all_pred_days = [r["pred_days"] for r in valid]
    all_bias = [r["mean_bias"] for r in valid]

    lines.append(f"\n## 整体\n")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 平均 MAPE | {np.mean(all_mape):.2f}% |")
    lines.append(f"| 中位数 MAPE | {np.median(all_mape):.2f}% |")
    lines.append(f"| 平均日间方向准确率 | {np.mean(all_dir):.1f}% |")
    lines.append(f"| 趋势方向正确率 (总涨跌方向) | {sum(all_trend)}/{len(all_trend)} = {sum(all_trend)/len(all_trend)*100:.1f}% |")
    lines.append(f"| 平均预测天数 | {np.mean(all_pred_days):.1f} |")
    lines.append(f"| 平均价差偏误 (预测-实际) | {np.mean(all_bias):.2f} |")

    # 按报告日期
    lines.append("\n## 按报告日期\n")
    lines.append("| 日期 | 样本 | MAPE | 方向准确率 | 趋势正确 | 平均偏误 |")
    lines.append("|------|------|------|-----------|---------|---------|")
    by_date = defaultdict(list)
    for r in valid:
        by_date[r["report_date"]].append(r)
    for d in sorted(by_date.keys()):
        rs = by_date[d]
        mape = np.mean([r["mape"] for r in rs])
        dir_acc = np.mean([r["direction_accuracy"] for r in rs])
        trend = sum(1 for r in rs if r["total_direction_correct"])
        bias = np.mean([r["mean_bias"] for r in rs])
        lines.append(f"| {d} | {len(rs)} | {mape:.1f}% | {dir_acc:.1f}% | {trend}/{len(rs)} | {bias:+.1f} |")

    # 按股票
    lines.append("\n## 按股票\n")
    lines.append("| 代码 | 名称 | 样本 | MAPE | 方向准确率 | 趋势正确 | 平均偏误 |")
    lines.append("|------|------|------|------|-----------|---------|---------|")
    by_stock = defaultdict(list)
    for r in valid:
        by_stock[r["code"]].append(r)
    for code in sorted(by_stock.keys()):
        rs = by_stock[code]
        name = rs[0]["name"]
        mape = np.mean([r["mape"] for r in rs])
        dir_acc = np.mean([r["direction_accuracy"] for r in rs])
        trend = sum(1 for r in rs if r["total_direction_correct"])
        bias = np.mean([r["mean_bias"] for r in rs])
        lines.append(f"| {code} | {name} | {len(rs)} | {mape:.1f}% | {dir_acc:.1f}% | {trend}/{len(rs)} | {bias:+.1f} |")

    # 按 MAPE 区间分布
    lines.append("\n## MAPE 分布\n")
    lines.append("| 区间 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    buckets = [("<10%", 10), ("10-20%", 20), ("20-30%", 30), ("30-50%", 50), (">=50%", float("inf"))]
    for label, threshold in buckets:
        prev = 0 if buckets.index((label, threshold)) == 0 else buckets[buckets.index((label, threshold)) - 1][1]
        count = sum(1 for m in all_mape if prev <= m < threshold)
        lines.append(f"| {label} | {count} | {count/len(all_mape)*100:.1f}% |")

    # 趋势方向正确率分布
    lines.append("\n## 趋势方向正确率\n")
    lines.append(f"- 正确: {sum(all_trend)} / {len(all_trend)} = {sum(all_trend)/len(all_trend)*100:.1f}%")
    lines.append(f"- 错误: {len(all_trend) - sum(all_trend)} / {len(all_trend)} = {(1 - sum(all_trend)/len(all_trend))*100:.1f}%")

    # 错误列表
    if errors:
        lines.append("\n## 失败记录\n")
        lines.append("| 报告日期 | 代码 | 名称 | 错误 |")
        lines.append("|----------|------|------|------|")
        for r in errors:
            lines.append(f"| {r['report_date']} | {r['code']} | {r.get('name', '?')} | {r['error']} |")

    # 参数
    lines.append("\n## 参数\n")
    lines.append(f"- 模型: {results[0].get('_model', 'Kronos-small')}")
    lines.append(f"- Lookback: {results[0].get('_lookback', '?')}")
    lines.append(f"- Pred len: {results[0].get('_pred_len', '?')}")
    param_modes = set(r.get("_params", "") for r in valid)
    if param_modes:
        lines.append(f"- 参数模式: {', '.join(sorted(param_modes))}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Kronos 回测")
    parser.add_argument("--reports", default="reports", help="报告目录 (默认 reports)")
    parser.add_argument("--out", default="evals/results", help="结果输出目录 (默认 evals/results)")
    parser.add_argument("--model", default="Kronos-small", help="模型名 (默认 Kronos-small)")
    parser.add_argument("--lookback", type=int, default=400, help="回溯天数 (默认 400)")
    parser.add_argument("--pred-len", type=int, default=15, help="预测天数 (默认 15)")
    parser.add_argument("--T", type=float, default=1.0, help="采样温度 (默认 1.0)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p 采样 (默认 0.9)")
    parser.add_argument("--device", default="cpu", help="设备 (cpu/cuda/mps)")
    parser.add_argument("--report-date", help="只跑指定日期的报告 (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=None, help="限制信号数 (调试用)")
    parser.add_argument("--per-stock", action="store_true", help="按个股差异化参数（覆盖 --lookback/--T/--top-p）")
    parser.add_argument("--sleep", type=float, default=0.3, help="每次取行情后 sleep 秒数")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    reports_dir = Path(args.reports).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载信号（获取报告日期 + 股票列表）
    if args.report_date:
        # 单日：只解析指定报告
        report_path = reports_dir / f"report_{args.report_date.replace('-', '')}.md"
        if not report_path.exists():
            print(f"❌ 报告不存在: {report_path}", file=sys.stderr)
            sys.exit(1)
        signals = parse_report(report_path)
        # 过滤掉无方向性信号（hold + neutral），保留至少有方向判断的
        logger.info("从 %s 解析到 %d 个信号", report_path.name, len(signals))
    else:
        signals = load_all_signals(reports_dir)
        logger.info("从 %s 抽取到 %d 个信号", reports_dir, len(signals))

    if not signals:
        print("❌ 未找到信号，退出。", file=sys.stderr)
        sys.exit(1)

    if args.limit:
        signals = signals[:args.limit]

    # 2. 去重：每个 (report_date, code) 只跑一次
    seen = set()
    unique_tasks = []
    for sig in signals:
        key = (sig.report_date, sig.code)
        if key in seen:
            continue
        seen.add(key)
        unique_tasks.append((sig.report_date, sig.code, sig.name))

    print(f"待跑任务: {len(unique_tasks)} 个 (去重后)", file=sys.stderr)

    # 3. 加载模型
    print(f"🤗 加载 Kronos 模型: {args.model} (device={args.device})...", file=sys.stderr)
    predictor = _load_kronos(args.model, args.device)

    # 4. 初始化数据源
    from data_provider.tencent_fetcher import TencentFetcher
    fetcher = TencentFetcher()

    # 5. 逐任务跑
    results = []
    t_start = time.time()
    for i, (report_date, code, name) in enumerate(unique_tasks):
        if args.verbose or (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            logger.info("[%d/%d] %s %s (%s) - 已耗时 %.1fs", i + 1, len(unique_tasks), report_date, name, code, elapsed)
        else:
            print(f"  [{i+1}/{len(unique_tasks)}] {report_date} {name}({code})", file=sys.stderr)

        r = evaluate_one(
            predictor, fetcher, code, name, report_date,
            args.lookback, args.pred_len, args.T, args.top_p,
            use_per_stock=args.per_stock,
        )
        r["_model"] = args.model
        r["_lookback"] = args.lookback
        r["_pred_len"] = args.pred_len
        results.append(r)

        time.sleep(args.sleep)

    # 6. 输出
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSONL（不含 _pred_series 以节省空间）
    jsonl_path = out_dir / f"kronos_outcomes_{ts}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            row = {k: v for k, v in r.items() if not k.startswith("_")}
            # 保留预测序列但不放进 JSONL 主记录（太大）
            row.pop("_pred_series", None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 详细 JSON（含预测序列，供后续分析）
    detail_path = out_dir / f"kronos_detail_{ts}.json"
    with detail_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 摘要
    summary_md = summarize(results)
    summary_path = out_dir / f"kronos_summary_{ts}.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    elapsed = time.time() - t_start
    print(f"\n✅ 完成! 耗时 {elapsed:.1f}s", file=sys.stderr)
    print(f"   逐条结果: {jsonl_path}", file=sys.stderr)
    print(f"   详细结果: {detail_path}", file=sys.stderr)
    print(f"   统计报告: {summary_path}", file=sys.stderr)
    print(file=sys.stderr)
    print(summary_md)


if __name__ == "__main__":
    main()