#!/usr/bin/env python3
"""Kronos 集成回测：跑 Kronos 预测并对比 LLM 信号命中率。

用法:
    cd daily_stock_analysis
    python -m evals.run_kronos_eval --out evals/results/kronos_evals
    python -m evals.run_kronos_eval --reports reports/ --horizon 5

输出:
    evals/results/kronos_evals/kronos_evals_<ts>.jsonl     逐条结果
    evals/results/kronos_evals/kronos_evals_<ts>.md        LLM vs Kronos 对比报告
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
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evals.extract_signals import load_all_signals, Signal  # noqa: E402
    from evals.kronos_params import get_stock_params  # noqa: E402
    from evals.kronos_eval import _load_kronos, _fetch_data, run_kronos_prediction, calc_metrics  # noqa: E402
else:
    from .extract_signals import load_all_signals, Signal
    from .kronos_params import get_stock_params
    from .kronos_eval import _load_kronos, _fetch_data, run_kronos_prediction, calc_metrics

# Kronos 路径
KRONOS_PATH = "/tmp/Kronos"
if KRONOS_PATH not in sys.path:
    sys.path.insert(0, KRONOS_PATH)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

logger = logging.getLogger("run_kronos_eval")

# =========================================================================
# 数据缓存
# =========================================================================
_data_cache: dict[str, pd.DataFrame] = {}


def _prefetch_all(fetcher, tasks: List[Tuple[str, str, str]], horizon: int):
    """预取所有股票行情数据，覆盖所有任务的最宽日期范围。"""
    # 按 code 聚合：计算每只股票需要的最早开始和最晚结束日期
    code_ranges: dict[str, Tuple[str, str]] = {}
    for report_date, code, _ in tasks:
        lb, _, _, _ = get_stock_params(code)
        start = pd.to_datetime(report_date) - timedelta(days=lb * 2 + 10)
        end = pd.to_datetime(report_date) + timedelta(days=horizon * 2 + 10)
        if code in code_ranges:
            ps, pe = code_ranges[code]
            code_ranges[code] = (
                min(ps, start.strftime("%Y-%m-%d")),
                max(pe, end.strftime("%Y-%m-%d")),
            )
        else:
            code_ranges[code] = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    for code, (start_str, end_str) in code_ranges.items():
        df = _fetch_data(fetcher, code, start_str, end_str)
        if not df.empty:
            _data_cache[code] = df


def _get_cached_data(code: str, report_date: str, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从缓存中获取基准日和窗口数据。

    Returns:
        (before_df, after_df) — 基准日之前（含）和之后的数据
    """
    df = _data_cache.get(code)
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    report_dt = pd.to_datetime(report_date)
    before = df[df["date"] <= report_dt].reset_index(drop=True)
    after = df[df["date"] > report_dt].reset_index(drop=True)
    return before, after


# =========================================================================
# 行情方向判定（与 outcomes.py 保持一致）
# =========================================================================
def _compute_direction_result(
    actual_df: pd.DataFrame, baseline_close: float, direction: str, horizon: int,
) -> Tuple[str, str, float, float]:
    """计算命中判定。

    Returns:
        (result_loose, result_strict, max_pct_change, end_pct_change)
    """
    window = actual_df.head(horizon)
    closes = window["close"].values
    pct_changes = [(c - baseline_close) / baseline_close for c in closes]

    # 宽松口径
    if direction == "long":
        max_pct = max(pct_changes) if pct_changes else 0.0
        result_loose = "hit" if max_pct > 0 else "miss"
    else:
        max_pct = min(pct_changes) if pct_changes else 0.0
        result_loose = "hit" if max_pct < 0 else "miss"

    # 严格口径
    end_pct = pct_changes[-1] if pct_changes else 0.0
    if len(closes) < horizon:
        result_strict = "pending"
    elif direction == "long":
        result_strict = "hit" if end_pct > 0 else "miss"
    else:
        result_strict = "hit" if end_pct < 0 else "miss"

    # 窗口未满 + miss → pending
    if len(closes) < horizon and result_loose == "miss":
        result_loose = "pending"
        result_strict = "pending"

    return result_loose, result_strict, max_pct, end_pct


# =========================================================================
# 核心：Kronos 预测 + 命中判定
# =========================================================================
def evaluate_kronos(
    predictor, code: str, name: str, report_date: str, horizon: int = 5,
) -> dict:
    """跑 Kronos 预测并做命中判定。

    Returns:
        dict 含 Kronos 预测结果和命中判定。
    """
    lb, T, tp, sc = get_stock_params(code)

    before, after = _get_cached_data(code, report_date, horizon)
    if before.empty or after.empty:
        return {"error": "取行情为空", "report_date": report_date, "code": code, "name": name}

    # 合并 before+after 构造完整 df 供 run_kronos_prediction 使用
    df = pd.concat([before, after], ignore_index=True)

    pred_df, actual_df, before_df, error = run_kronos_prediction(
        predictor, df, report_date, lb, horizon, T, tp, sc,
    )
    if error:
        return {"error": error, "report_date": report_date, "code": code, "name": name}

    # 预测指标
    metrics = calc_metrics(actual_df["close"].values, pred_df["close"].values[:len(actual_df)])
    last_price = float(before_df.iloc[-1]["close"])

    # Kronos 方向信号
    pred_total_return = metrics["pred_total_return_pct"]
    kronos_direction = "long" if pred_total_return > 0 else "short"

    # 基线价格
    baseline_close = float(before.iloc[-1]["close"])

    # 命中判定
    result_loose, result_strict, max_pct, end_pct = _compute_direction_result(
        after, baseline_close, kronos_direction, horizon,
    )

    # 实际方向（用于对比）
    actual_return = metrics["actual_total_return_pct"]
    actual_direction = "long" if actual_return > 0 else "short"

    return {
        "report_date": report_date,
        "code": code,
        "name": name,
        "last_price": last_price,
        "baseline_close": baseline_close,
        "kronos_direction": kronos_direction,
        "actual_direction": actual_direction,
        "pred_total_return": pred_total_return,
        "actual_total_return": actual_return,
        "result": result_loose,
        "result_strict": result_strict,
        "max_pct_change": max_pct,
        "end_pct_change": end_pct,
        "mape": metrics["mape"],
        "direction_accuracy": metrics["direction_accuracy"],
        "mean_bias": metrics["mean_bias"],
        "total_direction_correct": metrics["total_direction_correct"],
        "pred_days": metrics["pred_days"],
        "params": f"lb={lb},T={T},top_p={tp},sc={sc}",
    }


# =========================================================================
# LLM 信号处理
# =========================================================================
def _load_llm_outcomes(out_dir: Path) -> List[dict]:
    """从 evals 输出目录加载最新的 LLM 回测结果。"""
    jsonl_files = list(out_dir.glob("outcomes_*.jsonl"))
    if not jsonl_files:
        logger.warning("未找到 LLM 回测结果文件 (outcomes_*.jsonl)")
        return []
    latest = max(jsonl_files, key=lambda p: p.stat().st_mtime)
    logger.info("加载 LLM 回测结果: %s", latest.name)
    records = []
    with latest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _llm_effective_direction(signal_dict: dict) -> Optional[str]:
    """从信号 dict 中提取有效方向。"""
    dk = signal_dict.get("direction_kind", "")
    if dk in ("long", "short"):
        return dk
    ak = signal_dict.get("action_kind", "")
    if ak in ("long", "short"):
        return ak
    return None


def _build_llm_map(llm_records: List[dict]) -> dict:
    """构建 (report_date, code) -> {direction, result, result_strict} 映射。"""
    mapping = {}
    for r in llm_records:
        sig = r.get("signal", {})
        key = (sig.get("report_date", ""), sig.get("code", ""))
        direction = _llm_effective_direction(sig)
        if direction is None:
            continue
        mapping[key] = {
            "direction": direction,
            "result": r.get("result", "skipped"),
            "result_strict": r.get("result_strict", "skipped"),
            "action": sig.get("action", ""),
            "score": sig.get("score", 0),
            "name": sig.get("name", ""),
        }
    return mapping


# =========================================================================
# 汇总报告
# =========================================================================
def _fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _hit_rate(records: List[dict], strict: bool = False) -> Tuple[int, int, int, float]:
    """计算命中率。"""
    field = "result_strict" if strict else "result"
    hit = sum(1 for r in records if r.get(field) == "hit")
    miss = sum(1 for r in records if r.get(field) == "miss")
    pending = sum(1 for r in records if r.get(field) == "pending")
    judged = hit + miss
    rate = hit / judged if judged else 0.0
    return hit, miss, pending, rate


def generate_report(kronos_results: List[dict], llm_map: dict, horizon: int) -> str:
    """生成 LLM vs Kronos 对比报告。"""
    valid = [r for r in kronos_results if "error" not in r]
    errors = [r for r in kronos_results if "error" in r]

    # 匹配 LLM 数据
    matched = []
    for r in valid:
        key = (r["report_date"], r["code"])
        llm = llm_map.get(key)
        matched.append({**r, "llm": llm})

    # 仅统计 LLM 也有结果的（fair comparison）
    both = [m for m in matched if m["llm"] is not None]
    kronos_only = [m for m in matched if m["llm"] is None]

    lines = []
    lines.append(f"# Kronos vs LLM 信号对比报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"窗口: {horizon} 个交易日\n")
    lines.append(f"Kronos 模式: 按个股差异化参数 (evals.kronos_params)\n")
    lines.append(f"Kronos 总样本: {len(kronos_results)} | 成功: {len(valid)} | 失败: {len(errors)}\n")
    lines.append(f"LLM 匹配数: {len(both)} | Kronos 独有: {len(kronos_only)}\n")

    # 整体对比（仅两边都有结果的样本）
    if both:
        # Kronos 整体
        k_hit, k_miss, k_pend, k_rate = _hit_rate([m for m in both], strict=False)
        k_hit_s, k_miss_s, k_pend_s, k_rate_s = _hit_rate([m for m in both], strict=True)
        # LLM 整体
        l_hit = sum(1 for m in both if m["llm"]["result"] == "hit")
        l_miss = sum(1 for m in both if m["llm"]["result"] == "miss")
        l_pend = sum(1 for m in both if m["llm"]["result"] == "pending")
        l_judged = l_hit + l_miss
        l_rate = l_hit / l_judged if l_judged else 0.0
        l_hit_s = sum(1 for m in both if m["llm"]["result_strict"] == "hit")
        l_miss_s = sum(1 for m in both if m["llm"]["result_strict"] == "miss")
        l_pend_s = sum(1 for m in both if m["llm"]["result_strict"] == "pending")
        l_judged_s = l_hit_s + l_miss_s
        l_rate_s = l_hit_s / l_judged_s if l_judged_s else 0.0

        lines.append("\n## 整体命中率对比\n")
        lines.append("| 口径 | 模型 | hit | miss | pend | 有效 | 命中率 |")
        lines.append("|------|------|-----|------|------|------|--------|")
        lines.append(f"| 宽松 | LLM | {l_hit} | {l_miss} | {l_pend} | {l_judged} | {_fmt_pct(l_rate)} |")
        lines.append(f"| 宽松 | Kronos | {k_hit} | {k_miss} | {k_pend} | {k_hit + k_miss} | {_fmt_pct(k_rate)} |")
        lines.append(f"| 严格 | LLM | {l_hit_s} | {l_miss_s} | {l_pend_s} | {l_judged_s} | {_fmt_pct(l_rate_s)} |")
        lines.append(f"| 严格 | Kronos | {k_hit_s} | {k_miss_s} | {k_pend_s} | {k_hit_s + k_miss_s} | {_fmt_pct(k_rate_s)} |")

        # 按方向对比
        lines.append("\n## 按方向对比\n")
        lines.append("| 方向 | 样本 | LLM 命中率 | Kronos 命中率 | 差值 |")
        lines.append("|------|------|-----------|-------------|------|")
        for direction in ("long", "short"):
            k_dir = [m for m in both if m["kronos_direction"] == direction]
            l_dir = [m for m in both if m["llm"]["direction"] == direction]
            if not k_dir:
                continue
            _, _, _, k_r = _hit_rate(k_dir, strict=False)
            _, _, _, l_r = _hit_rate(l_dir, strict=False)
            diff = k_r - l_r
            n = len(k_dir)
            lines.append(f"| Kronos {direction} | {n} | {_fmt_pct(l_r)} | {_fmt_pct(k_r)} | {diff:+.1%} |")

        # 按股票对比
        lines.append("\n## 按股票对比\n")
        lines.append("| 代码 | 名称 | 样本 | LLM 宽松 | LLM 严格 | Kronos 宽松 | Kronos 严格 | MAPE | 偏误 |")
        lines.append("|------|------|------|---------|---------|-----------|-----------|------|------|")
        by_stock = defaultdict(list)
        for m in both:
            by_stock[m["code"]].append(m)
        for code in sorted(by_stock.keys()):
            rs = by_stock[code]
            name = rs[0]["name"]
            n = len(rs)
            # LLM 命中率（从 llm 子 dict 读取）
            l_hit = sum(1 for m in rs if m["llm"]["result"] == "hit")
            l_miss = sum(1 for m in rs if m["llm"]["result"] == "miss")
            l_pend = sum(1 for m in rs if m["llm"]["result"] == "pending")
            l_judged = l_hit + l_miss
            lr = l_hit / l_judged if l_judged else 0.0
            l_hit_s = sum(1 for m in rs if m["llm"]["result_strict"] == "hit")
            l_miss_s = sum(1 for m in rs if m["llm"]["result_strict"] == "miss")
            l_pend_s = sum(1 for m in rs if m["llm"]["result_strict"] == "pending")
            l_judged_s = l_hit_s + l_miss_s
            lr_s = l_hit_s / l_judged_s if l_judged_s else 0.0
            # Kronos 命中率
            kr, _, _, kr_r = _hit_rate([m for m in rs], strict=False)
            kr_s, _, _, kr_r_s = _hit_rate([m for m in rs], strict=True)
            avg_mape = np.mean([m["mape"] for m in rs])
            avg_bias = np.mean([m["mean_bias"] for m in rs])
            lines.append(
                f"| {code} | {name} | {n} | {_fmt_pct(lr)} | {_fmt_pct(lr_s)} "
                f"| {_fmt_pct(kr_r)} | {_fmt_pct(kr_r_s)} | {avg_mape:.1f}% | {avg_bias:+.1f} |"
            )

        # Kronos 预测指标汇总
        lines.append("\n## Kronos 预测指标\n")
        all_mape = [m["mape"] for m in matched]
        all_bias = [m["mean_bias"] for m in matched]
        all_dir = [m["direction_accuracy"] for m in matched]
        all_trend = [m["total_direction_correct"] for m in matched]
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 平均 MAPE | {np.mean(all_mape):.2f}% |")
        lines.append(f"| 中位数 MAPE | {np.median(all_mape):.2f}% |")
        lines.append(f"| 平均日间方向准确率 | {np.mean(all_dir):.1f}% |")
        lines.append(f"| 趋势方向正确率 | {sum(all_trend)}/{len(all_trend)} = {sum(all_trend)/len(all_trend)*100:.1f}% |")
        lines.append(f"| 平均偏误 | {np.mean(all_bias):.2f} |")

        # 方向一致率
        lines.append("\n## 方向一致率\n")
        same = sum(1 for m in both if m["kronos_direction"] == m["llm"]["direction"])
        lines.append(f"Kronos 与 LLM 方向一致: {same}/{len(both)} = {same/len(both)*100:.1f}%\n")
        # 谁的判断更准（仅当方向一致时）
        both_long = [m for m in both if m["kronos_direction"] == "long" and m["llm"]["direction"] == "long"]
        both_short = [m for m in both if m["kronos_direction"] == "short" and m["llm"]["direction"] == "short"]
        if both_long:
            lh, _, _, lr = _hit_rate(both_long, strict=False)
            kh, _, _, kr = _hit_rate(both_long, strict=False)
            lines.append(f"  共同看多: {len(both_long)} 个 — LLM {_fmt_pct(lr)} vs Kronos {_fmt_pct(kr)}")
        if both_short:
            lh, _, _, lr = _hit_rate(both_short, strict=False)
            kh, _, _, kr = _hit_rate(both_short, strict=False)
            lines.append(f"  共同看空: {len(both_short)} 个 — LLM {_fmt_pct(lr)} vs Kronos {_fmt_pct(kr)}")

    # 错误列表
    if errors:
        lines.append("\n## 失败记录\n")
        lines.append("| 报告日期 | 代码 | 名称 | 错误 |")
        lines.append("|----------|------|------|------|")
        for r in errors:
            lines.append(f"| {r['report_date']} | {r['code']} | {r.get('name', '?')} | {r['error']} |")

    return "\n".join(lines) + "\n"


# =========================================================================
# CLI
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Kronos 集成回测")
    parser.add_argument("--reports", default="reports", help="报告目录 (默认 reports)")
    parser.add_argument("--out", default="evals/results", help="结果输出目录 (默认 evals/results)")
    parser.add_argument("--horizon", type=int, default=5, help="回测窗口交易日 (默认 5)")
    parser.add_argument("--llm-out", default=None, help="LLM 回测结果目录 (默认与 --out 相同)")
    parser.add_argument("--sleep", type=float, default=0.3, help="取行情后 sleep 秒数")
    parser.add_argument("--limit", type=int, default=None, help="限制信号数 (调试用)")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    reports_dir = Path(args.reports).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_out_dir = Path(args.llm_out) if args.llm_out else out_dir

    # 1. 加载信号
    signals = load_all_signals(reports_dir)
    logger.info("从 %s 抽取到 %d 个信号", reports_dir, len(signals))
    if not signals:
        print("未找到信号，退出。", file=sys.stderr)
        sys.exit(1)

    # 2. 去重
    seen = set()
    tasks = []
    for sig in signals:
        key = (sig.report_date, sig.code)
        if key in seen:
            continue
        seen.add(key)
        tasks.append((sig.report_date, sig.code, sig.name))

    if args.limit:
        tasks = tasks[:args.limit]
        print(f"调试模式: 只跑前 {len(tasks)} 个任务", file=sys.stderr)

    print(f"待跑任务: {len(tasks)} 个 (去重后)", file=sys.stderr)

    # 3. 加载 Kronos 模型
    print("🤗 加载 Kronos 模型...", file=sys.stderr)
    predictor = _load_kronos()

    # 4. 初始化数据源
    from data_provider.tencent_fetcher import TencentFetcher
    fetcher = TencentFetcher()

    # 5. 预取数据（宽窗口覆盖所有任务）
    print("📥 预取行情数据...", file=sys.stderr)
    _prefetch_all(fetcher, tasks, args.horizon)
    print(f"   已缓存 {len(_data_cache)} 只股票", file=sys.stderr)

    # 6. 跑 Kronos 预测
    print(f"🔮 开始 Kronos 回测 (horizon={args.horizon})...", file=sys.stderr)
    results = []
    t_start = time.time()
    for i, (report_date, code, name) in enumerate(tasks):
        el = time.time() - t_start
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(tasks)}] {report_date} {name}({code}) - {el:.0f}s", file=sys.stderr)

        r = evaluate_kronos(predictor, code, name, report_date, args.horizon)
        results.append(r)
        time.sleep(args.sleep)

    elapsed = time.time() - t_start
    print(f"  ✅ 完成! 耗时 {elapsed:.1f}s", file=sys.stderr)

    # 7. 加载 LLM 结果
    llm_records = _load_llm_outcomes(llm_out_dir)
    llm_map = _build_llm_map(llm_records)
    if llm_records:
        print(f"   已加载 LLM 结果: {len(llm_records)} 条, 匹配 {len(llm_map)} 个", file=sys.stderr)

    # 8. 输出
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSONL
    jsonl_path = out_dir / f"kronos_evals_{ts}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            # 去掉内部字段
            row = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 报告
    summary_md = generate_report(results, llm_map, args.horizon)
    summary_path = out_dir / f"kronos_evals_{ts}.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    print(f"\n  逐条结果: {jsonl_path}", file=sys.stderr)
    print(f"  统计报告: {summary_path}", file=sys.stderr)
    print(file=sys.stderr)
    print(summary_md)


if __name__ == "__main__":
    import argparse
    main()