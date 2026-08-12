"""Evals CLI 入口：跑全量历史报告回测，输出命中率统计。

用法:
    cd daily_stock_analysis
    python -m evals.run_eval --horizon 5
    python -m evals.run_eval --horizon 5 --reports reports/ --out evals/results/

输出:
    evals/results/outcomes_<ts>.jsonl   逐条信号回测结果
    evals/results/summary_<ts>.md       分桶命中率统计表
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List

# 支持作为脚本直接运行（python evals/run_eval.py）和模块运行（python -m evals.run_eval）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evals.extract_signals import load_all_signals, Signal  # noqa: E402
    from evals.outcomes import compute_outcomes, Outcome  # noqa: E402
else:
    from .extract_signals import load_all_signals
    from .outcomes import compute_outcomes, Outcome


logger = logging.getLogger("evals")


def _bucket_by_score(score: int) -> str:
    if score >= 60:
        return ">=60"
    if score >= 45:
        return "45-59"
    return "<45"


def _hit_rate(results: List[Outcome], strict: bool = False) -> tuple:
    """返回 (hit, miss, pending, skipped, no_direction, judged, hit_rate)。

    strict=False 统计宽松口径 (result)，True 统计严格口径 (result_strict)。
    hit_rate 仅基于 hit+miss。
    """
    field = "result_strict" if strict else "result"
    hit = sum(1 for r in results if getattr(r, field) == "hit")
    miss = sum(1 for r in results if getattr(r, field) == "miss")
    pending = sum(1 for r in results if getattr(r, field) == "pending")
    skipped = sum(1 for r in results if getattr(r, field) == "skipped")
    no_dir = sum(1 for r in results if getattr(r, field) == "no_direction")
    judged = hit + miss
    rate = hit / judged if judged else 0.0
    return hit, miss, pending, skipped, no_dir, judged, rate


def _cmp_row(label: str, rs: List[Outcome], strict: bool) -> str:
    """宽松/严格对照表的一行。"""
    h, m, p, s, nd, j, rate = _hit_rate(rs, strict=strict)
    tag = "严格" if strict else "宽松"
    return f"| {label} | {tag} | {h} | {m} | {p} | {j} | {_fmt_pct(rate)} |"


def _fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def summarize(outcomes: List[Outcome], horizon: int) -> str:
    """生成 markdown 统计报告。"""
    lines: List[str] = []
    lines.append(f"# Evals 回测报告 (horizon={horizon} 交易日)\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"信号总数: {len(outcomes)}\n")

    h, m, p, s, nd, j, rate = _hit_rate(outcomes)
    lines.append(f"判定: hit={h} miss={m} pending={p} skipped={s} no_direction={nd} (有效 {j})\n")
    lines.append(f"整体命中率: **{_fmt_pct(rate)}**\n")

    # 宽松 vs 严格对照
    lines.append("\n## 宽松 vs 严格命中对照\n")
    lines.append("- 宽松: 窗口内任一收盘满足方向即 hit（已实现峰值/谷值）")
    lines.append("- 严格: 窗口末日收盘满足方向才 hit（期末方向）；窗口未满一律 pending\n")
    lines.append("| 分组 | 口径 | hit | miss | pend | 有效 | 命中率 |")
    lines.append("|------|------|-----|------|------|------|--------|")
    lines.append(_cmp_row("整体", outcomes, False))
    lines.append(_cmp_row("整体", outcomes, True))
    by_dir_cmp = defaultdict(list)
    for oc in outcomes:
        d = oc.signal.effective_direction
        if d in ("long", "short"):
            by_dir_cmp[d].append(oc)
    name_map = {"long": "看多 long", "short": "看空 short"}
    for d in ("long", "short"):
        rs = by_dir_cmp.get(d, [])
        if not rs:
            continue
        lines.append(_cmp_row(name_map[d], rs, False))
        lines.append(_cmp_row(name_map[d], rs, True))

    # 按方向
    lines.append("\n## 按方向\n")
    lines.append("| 方向 | hit | miss | pend | skipped | no_dir | 命中率 |")
    lines.append("|------|-----|------|------|---------|--------|--------|")
    by_dir = defaultdict(list)
    for oc in outcomes:
        d = oc.signal.effective_direction
        if d in ("long", "short"):
            by_dir[d].append(oc)
    for d in ("long", "short"):
        rs = by_dir.get(d, [])
        if not rs:
            continue
        h, m, p, s, nd, j, rate = _hit_rate(rs)
        lines.append(f"| {d} | {h} | {m} | {p} | {s} | {nd} | {_fmt_pct(rate)} |")

    # 按 action
    lines.append("\n## 按操作类型\n")
    lines.append("| 操作 | hit | miss | pend | skipped | no_dir | 命中率 |")
    lines.append("|------|-----|------|------|---------|--------|--------|")
    by_action = defaultdict(list)
    for oc in outcomes:
        by_action[oc.signal.action].append(oc)
    for action in sorted(by_action.keys()):
        h, m, p, s, nd, j, rate = _hit_rate(by_action[action])
        lines.append(f"| {action} | {h} | {m} | {p} | {s} | {nd} | {_fmt_pct(rate)} |")

    # 按评分区间
    lines.append("\n## 按评分区间\n")
    lines.append("| 区间 | hit | miss | pend | skipped | no_dir | 命中率 |")
    lines.append("|------|-----|------|------|---------|--------|--------|")
    by_score = defaultdict(list)
    for oc in outcomes:
        by_score[_bucket_by_score(oc.signal.score)].append(oc)
    for bucket in (">=60", "45-59", "<45"):
        rs = by_score.get(bucket, [])
        if not rs:
            continue
        h, m, p, s, nd, j, rate = _hit_rate(rs)
        lines.append(f"| {bucket} | {h} | {m} | {p} | {s} | {nd} | {_fmt_pct(rate)} |")

    # 按报告日期
    lines.append("\n## 按报告日期\n")
    lines.append("| 日期 | hit | miss | pend | skipped | no_dir | 命中率 |")
    lines.append("|------|-----|------|------|---------|--------|--------|")
    by_date = defaultdict(list)
    for oc in outcomes:
        by_date[oc.signal.report_date].append(oc)
    for d in sorted(by_date.keys()):
        h, m, p, s, nd, j, rate = _hit_rate(by_date[d])
        lines.append(f"| {d} | {h} | {m} | {p} | {s} | {nd} | {_fmt_pct(rate)} |")

    # skipped 原因 top
    skipped_reasons = defaultdict(int)
    for oc in outcomes:
        if oc.result == "skipped":
            # 归并同类原因
            r = oc.reason
            if "返回空行情" in r:
                r = "返回空行情"
            elif "取行情失败" in r:
                r = "取行情失败"
            elif "无交易日" in r:
                r = "报告日之后无交易日"
            skipped_reasons[r] += 1
    if skipped_reasons:
        lines.append("\n## Skipped 原因分布\n")
        lines.append("| 原因 | 数量 |")
        lines.append("|------|------|")
        for r, c in sorted(skipped_reasons.items(), key=lambda x: -x[1]):
            lines.append(f"| {r} | {c} |")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="跑历史决策报告回测 evals")
    parser.add_argument("--reports", default="reports", help="报告目录 (默认 reports)")
    parser.add_argument("--horizon", type=int, default=5, help="回测窗口交易日 (默认 5)")
    parser.add_argument("--out", default="evals/results", help="结果输出目录")
    parser.add_argument("--sleep", type=float, default=0.5, help="每次取行情后 sleep 秒数")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个信号(调试用)")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    reports_dir = Path(args.reports).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 抽信号
    signals = load_all_signals(reports_dir)
    logger.info("从 %s 抽取到 %d 个信号", reports_dir, len(signals))
    if not signals:
        print(f"未在 {reports_dir} 找到 report_*.md 信号，退出。", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        signals = signals[: args.limit]
        print(f"调试模式: 只跑前 {len(signals)} 个信号", file=sys.stderr)

    # 2. 取行情并判定
    from data_provider.akshare_fetcher import AkshareFetcher

    fetcher = AkshareFetcher()
    print(f"开始回测 {len(signals)} 个信号，horizon={args.horizon}...", file=sys.stderr)
    outcomes = compute_outcomes(signals, fetcher, horizon=args.horizon, sleep_per_call=args.sleep)

    # 3. 输出
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"outcomes_{ts}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for oc in outcomes:
            f.write(json.dumps(oc.to_dict(), ensure_ascii=False) + "\n")

    summary_path = out_dir / f"summary_{ts}.md"
    summary_path.write_text(summarize(outcomes, args.horizon), encoding="utf-8")

    print(f"\n逐条结果: {jsonl_path}", file=sys.stderr)
    print(f"统计报告: {summary_path}", file=sys.stderr)
    print(file=sys.stderr)
    print(summarize(outcomes, args.horizon))


if __name__ == "__main__":
    main()
