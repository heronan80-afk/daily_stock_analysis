"""从已落盘的 outcomes jsonl 离线重算严格口径，重生成 summary。

历史行情不变，window_closes/baseline_close 已在 jsonl 里，
无需再调 akshare。复用 run_eval.summarize 输出含「宽松 vs 严格对照」的报告。
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import summarize  # noqa: E402

HORIZON = 5
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "evals/results/outcomes_20260728_213017.jsonl"


def effective_direction(sig: dict) -> str:
    dk = sig["direction_kind"]
    return dk if dk in ("long", "short") else sig["action_kind"]


def recompute_strict(rec: dict) -> tuple:
    """返回 (end_pct_change, result_strict)。逻辑与 outcomes.py 一致。"""
    res = rec["result"]
    if res in ("skipped", "no_direction"):
        return None, res
    closes = rec["window_closes"]
    baseline = rec["baseline_close"]
    if not closes or baseline is None:
        return None, "skipped"
    end_pct = (closes[-1] - baseline) / baseline
    direction = effective_direction(rec["signal"])
    if len(closes) < HORIZON:
        return end_pct, "pending"
    if direction == "long":
        return end_pct, ("hit" if end_pct > 0 else "miss")
    if direction == "short":
        return end_pct, ("hit" if end_pct < 0 else "miss")
    return end_pct, "no_direction"


def to_ns(rec: dict) -> SimpleNamespace:
    sig = SimpleNamespace(
        report_date=rec["signal"]["report_date"],
        action=rec["signal"]["action"],
        score=rec["signal"]["score"],
        effective_direction=effective_direction(rec["signal"]),
    )
    end_pct, strict = recompute_strict(rec)
    return SimpleNamespace(
        signal=sig,
        result=rec["result"],
        result_strict=strict,
        end_pct_change=end_pct,
        reason=rec.get("reason", ""),
    )


def main():
    recs = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    outcomes = [to_ns(r) for r in recs]

    # 不变量校验：严格 hit => 宽松 hit（期末满足 => 窗口内任一满足）。
    # 反向不成立：宽松 hit 但严格 miss 正是"冲高回落/探底回升"的规则红利 case。
    bad = 0
    for r, ns in zip(recs, outcomes):
        if ns.result_strict == "hit" and ns.result != "hit":
            bad += 1
    print(f"records={len(recs)} 不一致(严格hit但宽松非hit)={bad}", file=sys.stderr)
    assert bad == 0, "严格 hit 必须蕴含宽松 hit，校验失败"

    md = summarize(outcomes, HORIZON)
    out = ROOT / "evals/results/summary_20260728_213017.md"
    out.write_text(md, encoding="utf-8")
    print(f"写入 {out}", file=sys.stderr)
    print(md)


if __name__ == "__main__":
    main()
