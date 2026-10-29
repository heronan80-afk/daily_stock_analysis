#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补跑上周大盘复盘（一次性操作脚本）。

背景：2026-09-07 ~ 2026-09-11 的大盘复盘因 LLM 额度耗尽，保存成了只有数据表格、
缺分析正文的降级模板（~2.5KB）。本脚本从这些降级报告中解析出当天完整数据，
重建 MarketOverview，调用当前配置的 LLM 重新生成完整报告，并覆盖同名文件。

用法（在项目根目录）:
    venv/bin/python backfill_market_review.py                 # 补跑全部 5 天
    venv/bin/python backfill_market_review.py --date 20260911 # 只补跑某一天
    venv/bin/python backfill_market_review.py --check         # 只做数据重建校验，不调用 LLM
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
)
logger = logging.getLogger("backfill_market_review")

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# (日期, 文件名日期串)
TARGET_DATES = [
    ("2026-09-07", "20260907"),
    ("2026-09-08", "20260908"),
    ("2026-09-09", "20260909"),
    ("2026-09-10", "20260910"),
    ("2026-09-11", "20260911"),
]

INDEX_CODE = {
    "上证指数": "000001",
    "深证成指": "399001",
    "创业板指": "399006",
    "科创50": "000688",
    "上证50": "000016",
    "沪深300": "000300",
}


def _opt_number(value: str) -> float:
    """把 '123.45' / 'N/A' / '-' 转成 float / 0.0。"""
    value = value.strip().replace(",", "")
    if value in ("", "N/A", "-", "—"):
        return 0.0
    return float(value)


def _parse_sector_rows(text: str, heading: str) -> list[dict]:
    """解析形如 '| 1 | 板块名 | +1.09% |' 的表格行（head 之后的连续数据行）。"""
    idx = text.find(heading)
    if idx < 0:
        return []
    rows = []
    for line in text[idx:].splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if rows:
                break
            continue
        m = re.match(r"\|\s*\d+\s*\|\s*(.+?)\s*\|\s*([+\-]?\d+(?:\.\d+)?)%\s*\|", stripped)
        if m:
            rows.append({"name": m.group(1).strip(), "change_pct": float(m.group(2))})
    return rows


def parse_report_markdown(text: str, date: str) -> "object":
    """从降级报告 markdown 文本中重建 MarketOverview。"""
    from src.market_analyzer import MarketIndex, MarketOverview

    def find_int(pattern: str, group: int) -> int:
        m = re.search(pattern, text)
        return int(m.group(group)) if m else 0

    up = find_int(r"上涨/下跌/平盘\s*\|\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", 1)
    down = find_int(r"上涨/下跌/平盘\s*\|\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", 2)
    flat = find_int(r"上涨/下跌/平盘\s*\|\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", 3)
    limit_up = find_int(r"涨停/跌停\s*\|\s*(\d+)\s*/\s*(\d+)", 1)
    limit_down = find_int(r"涨停/跌停\s*\|\s*(\d+)\s*/\s*(\d+)", 2)

    total_amount = 0.0
    m = re.search(r"两市成交额\s*\|\s*([\d,]+)\s*亿", text)
    if m:
        total_amount = float(m.group(1).replace(",", ""))

    # 指数结构表
    indices = []
    for line in text.splitlines():
        m = re.match(
            r"\|\s*(上证指数|深证成指|创业板指|科创50|上证50|沪深300)\s*"
            r"\|\s*([\d.]+)\s*"
            r"\|\s*(?:🔴|🟢|⚪)\s*([+\-]?[\d.]+)%\s*"
            r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*"
            r"\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|",
            line.strip(),
        )
        if not m:
            continue
        name, current_s, pct_s, o_s, h_s, l_s, amp_s, amt_s = m.groups()
        change_pct = float(pct_s)
        current = float(current_s)
        prev_close = current / (1 + change_pct / 100.0)
        indices.append(
            MarketIndex(
                code=INDEX_CODE[name],
                name=name,
                current=current,
                change=current - prev_close,
                change_pct=change_pct,
                open=_opt_number(o_s),
                high=_opt_number(h_s),
                low=_opt_number(l_s),
                prev_close=prev_close,
                volume=0.0,
                amount=float(amt_s) * 1e8,  # 报告单位为亿，内部单位为元
                amplitude=_opt_number(amp_s),
            )
        )

    top_sectors = _parse_sector_rows(text, "行业板块领涨")
    bottom_sectors = _parse_sector_rows(text, "行业板块领跌")

    return MarketOverview(
        date=date,
        indices=indices,
        up_count=up,
        down_count=down,
        flat_count=flat,
        limit_up_count=limit_up,
        limit_down_count=limit_down,
        total_amount=total_amount,
        top_sectors=top_sectors,
        bottom_sectors=bottom_sectors,
        top_concepts=[],
        bottom_concepts=[],
    )


def parse_report_file(path: Path, date: str) -> "object":
    """从降级报告文件中重建 MarketOverview。"""
    return parse_report_markdown(path.read_text(encoding="utf-8"), date)


def load_source_markdown_from_db(date: str, date_str: str) -> str:
    """从 analysis_history 读取当日原始大盘复盘 markdown。

    报告文件可能已被补跑覆盖（丢数据），但 DB 里保留着原始降级版全文
    （raw_result -> raw_response），是补跑的可信数据源。
    """
    import json
    import sqlite3

    db_path = Path(__file__).resolve().parent / "data" / "stock_analysis.db"
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT raw_result FROM analysis_history "
            "WHERE code='MARKET' AND analysis_summary LIKE ? "
            "ORDER BY created_at DESC LIMIT 1",
            (f"{date} 大盘复盘%",),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise FileNotFoundError(f"DB 中未找到 {date} 的大盘复盘原始记录")

    payload = json.loads(row[0])
    raw_response = payload.get("raw_response") or ""
    if not raw_response.strip():
        raise FileNotFoundError(f"DB 中 {date} 的原始报告为空")
    return raw_response


def _patch_section_patterns() -> None:
    """放宽大盘报告章节注入正则，容忍标题中 emoji 出现在序号前（LLM 偶尔这么写）。

    仅影响本脚本进程内加载的模块，不改动仓库代码。
    """
    import src.market_analyzer as market_analyzer_module

    for name in ("_CHINESE_SECTION_PATTERNS", "_ENGLISH_SECTION_PATTERNS"):
        patterns = getattr(market_analyzer_module, name)
        # 将开头 "###\s*"（字面字符 ###\s*）放宽为 "###[^\n]*?"，允许 ### 后带 emoji
        setattr(
            market_analyzer_module,
            name,
            {k: v.replace("###\\s*", "###[^\\n]*?") for k, v in patterns.items()},
        )


def validate_overview(overview: "object", source_text: str) -> list[str]:
    """用报告里标注的盘面信号分数核对重建数据是否一致。"""
    issues = []
    m = re.search(r"\*\*盘面信号\*\*：(\d+)/100（(.+?)，(.+?)）", source_text)
    if not m:
        return ["未在报告中找到盘面信号标注，跳过一致性校验"]

    from src.market_analyzer import MarketAnalyzer
    from src.config import get_config

    analyzer = MarketAnalyzer(region="cn", config=get_config())
    light = analyzer.build_market_light_snapshot(overview)
    expected_score, expected_label = int(m.group(1)), m.group(2)
    if light["score"] != expected_score:
        issues.append(
            f"盘面信号分不一致：报告 {expected_score}/{expected_label}，重建后 {light['score']}/{light['temperature_label']}"
        )
    else:
        logger.info(
            "盘面信号分校验一致: %s/100 (%s)",
            light["score"],
            light["temperature_label"],
        )
    if not overview.indices:
        issues.append("未解析到任何指数数据")
    if not (overview.up_count or overview.down_count):
        issues.append("未解析到涨跌家数")
    return issues


def generate_and_save(date: str, date_str: str, check_only: bool) -> bool:
    path = REPORTS_DIR / f"market_review_{date_str}.md"

    from src.config import get_config
    from src.market_analyzer import MarketAnalyzer
    from src.core.market_review_runtime import build_market_review_runtime

    # 优先从 DB 取原始降级版全文（文件可能已被补跑覆盖），失败再退回文件
    source_text = None
    try:
        source_text = load_source_markdown_from_db(date, date_str)
        logger.info("[%s] 原始数据来源: analysis_history(DB)", date)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] DB 读取原始报告失败，退回文件: %s", date, exc)
        if not path.exists():
            logger.error("[%s] 文件与 DB 均无可用数据，跳过", date)
            return False
        source_text = path.read_text(encoding="utf-8")

    overview = parse_report_markdown(source_text, date)
    logger.info(
        "[%s] 重建数据: 指数=%d 上涨=%d 下跌=%d 平=%d 涨停=%d 跌停=%d 成交额=%.0f亿 领涨板块=%d 领跌板块=%d",
        date,
        len(overview.indices),
        overview.up_count,
        overview.down_count,
        overview.flat_count,
        overview.limit_up_count,
        overview.limit_down_count,
        overview.total_amount,
        len(overview.top_sectors),
        len(overview.bottom_sectors),
    )

    issues = validate_overview(overview, source_text)
    for issue in issues:
        logger.warning("[%s] 校验问题: %s", date, issue)

    if check_only:
        return True

    _patch_section_patterns()

    config = get_config()
    _, analyzer, _ = build_market_review_runtime(config)
    market_analyzer = MarketAnalyzer(analyzer=analyzer, region="cn", config=config)

    logger.info("[%s] 开始 LLM 生成正文 ...", date)
    # 不传新闻：历史日期无法回填当天新闻，避免用当前新闻造成时间错位
    report = None
    for attempt in range(1, 4):
        report = market_analyzer.generate_market_review(overview, news=[])
        if not report or not report.strip():
            logger.warning("[%s] LLM 第 %d 次返回空报告，重试", date, attempt)
            continue
        # 生成后校验：盘面总览表格与指数表格必须已注入，否则重试
        if "上涨/下跌/平盘" in report and "| 上证指数 |" in report:
            break
        logger.warning(
            "[%s] LLM 第 %d 次生成缺少数据表格（盘面/指数），重试",
            date,
            attempt,
        )
        report = None

    if not report or not report.strip():
        logger.error("[%s] LLM 连续多次生成失败，放弃覆盖", date)
        return False

    content = f"# 🎯 大盘复盘\n\n{report.strip()}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("[%s] 已覆盖 %s (%d 字符)", date, path, len(content))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="补跑上周大盘复盘")
    parser.add_argument("--date", type=str, default=None, help="如 20260911，缺省补跑全部 5 天")
    parser.add_argument("--check", action="store_true", help="只做数据重建校验，不调用 LLM")
    args = parser.parse_args()

    dates = TARGET_DATES
    if args.date:
        dates = [d for d in TARGET_DATES if d[1] == args.date]
        if not dates:
            logger.error("未知日期: %s（可用: %s）", args.date, [d[1] for d in TARGET_DATES])
            return 2

    ok = True
    for date, date_str in dates:
        try:
            ok = generate_and_save(date, date_str, args.check) and ok
        except Exception as exc:  # noqa: BLE001 - 单天失败不阻断后续
            logger.exception("[%s] 补跑失败: %s", date, exc)
            ok = False

    logger.info("补跑完成: %s", "全部成功" if ok else "存在失败，请查看上方日志")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
