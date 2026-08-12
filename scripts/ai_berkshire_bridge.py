#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai-berkshire bridge — deep research integration for daily_stock_analysis

Architecture:
  daily_stock_analysis (sentinel)  ──signal──▶  ai-berkshire (deep research toolbox)

Role:
  - Daily stock analysis detects stocks with strong buy/sell signals
  - This bridge auto-invokes ai-berkshire's investment-research skill for those stocks
  - Research reports are saved to ai-berkshire's reports/ directory
  - Results are pushed to Feishu alongside the daily report

Usage:
  # From within daily_stock_analysis project:
  python scripts/ai_berkshire_bridge.py --stock 600519 --stock-name 贵州茅台

  # With analysis results JSON:
  python scripts/ai_berkshire_bridge.py --results reports/latest_results.json

  # Dry run (show what would be researched without executing):
  python scripts/ai_berkshire_bridge.py --results reports/latest_results.json --dry-run

Environment:
  AI_BERKSHIRE_PATH    Path to ai-berkshire repo (default: ~/projects/ai-berkshire)
  AI_BERKSHIRE_SKILL   Claude skill to invoke (default: investment-research)
  FEISHU_WEBHOOK_URL   Feishu webhook URL for notification (optional)
  BRIDGE_SIGNAL_MIN    Minimum signal strength to trigger research (default: 70)
  CLAUDE_CLI_PATH      Path to Claude CLI (default: auto-detect)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai_berkshire_bridge")

# ── Configuration defaults ────────────────────────────────────────────

DEFAULT_AI_BERKSHIRE_PATH = os.path.expanduser("~/projects/ai-berkshire")
DEFAULT_SKILL = "investment-research"
DEFAULT_SIGNAL_MIN = 60  # Research score >= 60 (bullish) or <= 40 (bearish)
DEFAULT_MAX_SIGNALS = 6  # Cap research per run - each stock invokes Claude CLI
DEFAULT_CLAUDE_TIMEOUT = 300  # 5 minutes max per research call

# Action keywords - matched as substrings so daily_stock_analysis's compound
# advice (e.g. "减仓/卖出", "观望/减仓") still triggers. daily emits "减仓"
# for reduce, NOT "减持" - both are accepted here (historical bug source).
BULLISH_ACTION_KEYWORDS = ("buy", "add", "strong_buy", "买入", "增持", "加仓")
BEARISH_ACTION_KEYWORDS = ("sell", "reduce", "strong_sell", "卖出", "减持", "减仓")
DEFAULT_CLAUDE_CLI = "claude"

# ── Helpers ───────────────────────────────────────────────────────────


def _find_claude_cli() -> str:
    """Locate the Claude CLI binary."""
    candidates = ["claude", "claude-code"]
    for cmd in candidates:
        try:
            result = subprocess.run(
                ["which", cmd], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    # Fallback to common install locations
    fallbacks = [
        os.path.expanduser("~/.npm-global/bin/claude"),
        "/usr/local/bin/claude",
        "/opt/homebrew/bin/claude",
    ]
    for path in fallbacks:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return DEFAULT_CLAUDE_CLI


def _read_feishu_webhook() -> Optional[str]:
    """Read FEISHU_WEBHOOK_URL from env or daily_stock_analysis .env."""
    url = os.environ.get("FEISHU_WEBHOOK_URL")
    if url:
        return url

    # Try reading from .env
    env_path = Path(os.environ.get("DSA_PROJECT_DIR", os.getcwd())) / ".env"
    if env_path.exists():
        try:
            from dotenv import dotenv_values
            values = dotenv_values(str(env_path))
            return values.get("FEISHU_WEBHOOK_URL") or None
        except ImportError:
            # Manual parsing
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("FEISHU_WEBHOOK_URL="):
                    return line.split("=", 1)[1].strip().strip('"\'')
        except Exception:
            pass
    return None


def _send_feishu_notification(webhook_url: str, title: str, content: str) -> bool:
    """Send a markdown message to Feishu via webhook."""
    import urllib.request

    # Truncate if too long (Feishu webhook limit ~30000 bytes)
    content_bytes = content.encode("utf-8")
    if len(content_bytes) > 25000:
        content = content[:20000] + "\n\n...（内容过长已截断，完整报告请查看本地文件）"

    payload = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "blue",
            },
            "elements": [
                {"tag": "markdown", "content": content},
                {
                    "tag": "hr",
                },
                {
                    "tag": "note",
                    "elements": [
                        {
                            "tag": "plain_text",
                            "content": f"ai-berkshire bridge · {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        }
                    ],
                },
            ],
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if result.get("StatusCode") == 0 or result.get("code") == 0:
                logger.info("Feishu notification sent successfully")
                return True
            else:
                logger.warning("Feishu notification returned: %s", result)
                return False
    except Exception as e:
        logger.error("Feishu notification failed: %s", e)
        return False


def _send_feishu_simple_message(webhook_url: str, text: str) -> bool:
    """Send a simple text message to Feishu."""
    import urllib.request

    payload = {
        "msg_type": "text",
        "content": {"text": text},
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("StatusCode") == 0 or result.get("code") == 0
    except Exception as e:
        logger.error("Feishu simple message failed: %s", e)
        return False


# ── Stock signal detection ────────────────────────────────────────────


def detect_signal_stocks(
    analysis_results: List[Dict[str, Any]],
    min_score: int = DEFAULT_SIGNAL_MIN,
    max_signals: int = DEFAULT_MAX_SIGNALS,
) -> List[Dict[str, Any]]:
    """Filter analysis results to find stocks with strong signals.

    A "strong signal" means:
    - Score >= min_score AND action looks bullish (买入/增持/加仓/...), OR
    - Score <= (100 - min_score) AND action looks bearish (卖出/减仓/减持/...), OR
    - Score >= 80 with no explicit action (treated as bullish).

    Action matching is substring-based, so daily_stock_analysis's compound
    advice (e.g. "减仓/卖出", "观望/减仓") still triggers. daily emits "减仓"
    for reduce, not "减持" - both are accepted here.

    At most ``max_signals`` stocks are returned (0 = unlimited), prioritised
    by how far the score sits from the neutral 50 so a busy day does not fan
    out into an unbounded number of Claude research calls.

    Returns list of dicts with keys: code, name, score, action, signal_type, reason
    """
    signals = []

    for result in analysis_results:
        code = result.get("stock_code", result.get("code", ""))
        name = result.get("stock_name", result.get("name", code))
        score = result.get("score", result.get("signal_score", 50))
        action = result.get(
            "action",
            result.get("operation_advice", result.get("decision_type", "")),
        )

        # Normalize score to int
        try:
            score = int(score)
        except (ValueError, TypeError):
            score = 50

        # Normalize action
        action_str = str(action).lower() if action else ""

        # Check if this stock has a strong signal (substring match on action)
        is_strong = False
        signal_type = ""

        if score >= min_score and any(k in action_str for k in BULLISH_ACTION_KEYWORDS):
            is_strong = True
            signal_type = "bullish"
        elif score <= (100 - min_score) and any(k in action_str for k in BEARISH_ACTION_KEYWORDS):
            is_strong = True
            signal_type = "bearish"
        elif score >= 80 and not action_str:
            # High score even without explicit action
            is_strong = True
            signal_type = "bullish"

        if is_strong:
            reason = result.get("reason", result.get("signal_reasons", ""))
            if isinstance(reason, list):
                reason = "; ".join(reason)

            signals.append({
                "code": code,
                "name": name,
                "score": score,
                "action": action_str,
                "signal_type": signal_type,
                "reason": reason,
                "raw_result": result,
            })

    # Dedup by code: keep the most extreme signal per stock. A stock may
    # appear multiple times if the day's analysis ran more than once.
    seen: Dict[str, Dict[str, Any]] = {}
    for s in signals:
        c = s["code"]
        if c not in seen or abs(s["score"] - 50) > abs(seen[c]["score"] - 50):
            seen[c] = s
    if len(seen) != len(signals):
        logger.info("Deduped %d signals to %d unique stocks", len(signals), len(seen))
    signals = list(seen.values())

    # Cap to max_signals: most extreme (furthest from neutral 50) first.
    if max_signals and len(signals) > max_signals:
        signals.sort(key=lambda s: abs(s["score"] - 50), reverse=True)
        dropped = signals[max_signals:]
        logger.info(
            "Capping to %d strongest signals, dropped %d less extreme: %s",
            max_signals, len(dropped),
            ", ".join(f"{s['name']}({s['score']})" for s in dropped),
        )
        signals = signals[:max_signals]

    return signals


def _load_results_from_db(results_path: str) -> List[Dict[str, Any]]:
    """Fallback: read today's analysis results from the daily_stock_analysis SQLite DB.

    daily-stock-analysis stores per-stock results in the ``analysis_history`` table
    (code/name/sentiment_score/operation_advice/analysis_summary) but does NOT write a
    ``reports/*.json`` file. This bridges that gap so the bridge can run without a JSON
    artifact. The DB is resolved via ``DSA_DB_PATH`` env var, then ``<cwd>/data/stock_analysis.db``,
    then a path relative to the ``--results`` directory.
    """
    import sqlite3

    db_path = os.environ.get("DSA_DB_PATH") or os.path.join(
        os.getcwd(), "data", "stock_analysis.db"
    )
    if not os.path.isfile(db_path):
        alt = os.path.join(
            os.path.dirname(os.path.abspath(results_path)), "..", "data", "stock_analysis.db"
        )
        db_path = alt if os.path.isfile(alt) else db_path
    if not os.path.isfile(db_path):
        logger.warning("analysis_history DB not found (looked for %s)", db_path)
        return []

    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT code, name, sentiment_score, operation_advice, analysis_summary "
            "FROM analysis_history "
            "WHERE code != 'MARKET' AND created_at >= date('now','localtime') "
            "ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
    except Exception as e:
        logger.error("Failed to query analysis_history: %s", e)
        return []

    seen: set = set()
    results: List[Dict[str, Any]] = []
    for code, name, score, advice, summary in rows:
        if not code or code in seen:
            continue
        seen.add(code)
        try:
            score_int = int(score) if score is not None else 50
        except (ValueError, TypeError):
            score_int = 50
        results.append({
            "code": code,
            "name": name or code,
            "score": score_int,
            "operation_advice": advice or "",
            "reason": (summary or "")[:500],
        })
    return results


def parse_analysis_results(results_path: str) -> List[Dict[str, Any]]:
    """Parse analysis results from a JSON file or directory."""
    path = Path(results_path)

    if not path.exists():
        logger.error("Results path not found: %s", results_path)
        return []

    if path.is_dir():
        # Look for the latest JSON result file
        json_files = sorted(path.glob("*.json"), key=os.path.getmtime, reverse=True)
        if not json_files:
            logger.warning(
                "No JSON files in %s — falling back to SQLite analysis_history",
                results_path,
            )
            db_results = _load_results_from_db(results_path)
            if db_results:
                logger.info(
                    "Loaded %d results from DB (today's analysis_history)",
                    len(db_results),
                )
                return db_results
            logger.error(
                "No JSON files in %s and no today records in analysis_history DB",
                results_path,
            )
            return []
        path = json_files[0]
        logger.info("Using latest results file: %s", path)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError) as e:
        logger.error("Failed to parse results file: %s", e)
        return []

    # Handle different formats
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Try common keys for result lists
        for key in ("results", "stocks", "analysis_results", "items", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # Single result
        return [data]

    return []


# ── AI Berkshire invocation ───────────────────────────────────────────


def resolve_question_for_stock(
    stock: Dict[str, Any],
    skill: str = DEFAULT_SKILL,
) -> str:
    """Build the research question based on signal type and stock info."""
    code = stock["code"]
    name = stock["name"]
    signal_type = stock.get("signal_type", "bullish")
    reason = stock.get("reason", "")

    if signal_type == "bullish":
        base = (
            f"对股票 {code} {name} 进行系统性投资研究分析。"
            f"该股票今日分析评分{stock['score']}分，信号为买入/看多。"
        )
        if reason:
            base += f"\n触发原因：{reason}\n"
        base += (
            "\n请重点分析：\n"
            "1. 当前估值是否合理（PE/PB/PS band分析）\n"
            "2. 基本面是否支持继续上涨（收入/利润趋势、ROE、现金流）\n"
            "3. 核心风险是什么（竞争、估值、政策、技术）\n"
            "4. 安全边际在哪里\n"
            "5. 给出明确的投资建议：买入/持有/卖出及理由"
        )
    else:
        base = (
            f"对股票 {code} {name} 进行系统性投资研究分析。"
            f"该股票今日分析评分{stock['score']}分，信号为卖出/看空。"
        )
        if reason:
            base += f"\n触发原因：{reason}\n"
        base += (
            "\n请重点分析：\n"
            "1. 基本面恶化程度（收入/利润趋势、ROE、负债率变化）\n"
            "2. 估值是否仍然合理\n"
            "3. 下跌空间有多大\n"
            "4. 是否有逆转可能\n"
            "5. 给出明确的投资建议：卖出/持有/等待及理由"
        )

    return base


def invoke_ai_berkshire_research(
    stock_code: str,
    stock_name: str,
    question: str,
    skill: str = DEFAULT_SKILL,
    ai_berkshire_path: str = DEFAULT_AI_BERKSHIRE_PATH,
    claude_cli: str = DEFAULT_CLAUDE_CLI,
    timeout: int = DEFAULT_CLAUDE_TIMEOUT,
) -> Tuple[bool, str]:
    """Invoke ai-berkshire's investment-research skill via Claude CLI.

    Returns:
        Tuple of (success: bool, report_text: str)
    """
    # Verify paths
    berkshire_dir = Path(ai_berkshire_path)
    if not berkshire_dir.exists():
        return False, f"ai-berkshire repo not found at {ai_berkshire_path}"

    # Verify Claude CLI
    claude_path = _find_claude_cli() if claude_cli == DEFAULT_CLAUDE_CLI else claude_cli
    if not shutil_which(claude_path):
        return False, f"Claude CLI not found: {claude_path}"

    # Build the prompt — reference the skill at the start
    prompt = f"/{skill}\n\n{question}"

    logger.info(
        "Invoking ai-berkshire (%s) for %s (%s)...",
        skill, stock_code, stock_name,
    )

    cmd = [
        claude_path,
        "-p", prompt,
        "--print",
        "--no-session-persistence",
        "--add-dir", str(berkshire_dir),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                **os.environ,
                "CLAUDE_CODE_SIMPLE": "1",
            },
        )

        if result.returncode != 0:
            stderr = result.stderr.strip() or "Unknown error"
            logger.warning(
                "Claude CLI returned code %d for %s: %s",
                result.returncode, stock_code, stderr[:500],
            )
            # Return partial output if any
            output = result.stdout.strip()
            if output:
                return True, output
            return False, f"Claude CLI failed (exit={result.returncode}): {stderr[:500]}"

        report = result.stdout.strip()
        if not report:
            return False, f"Claude CLI returned empty output for {stock_code}"

        return True, report

    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI timed out (%ds) for %s", timeout, stock_code)
        return False, f"Research timed out after {timeout}s for {stock_code}"
    except FileNotFoundError as e:
        return False, f"Claude CLI not found: {e}"
    except Exception as e:
        logger.error("Failed to invoke Claude CLI for %s: %s", stock_code, e)
        return False, str(e)


def shutil_which(cmd: str) -> Optional[str]:
    """Check if a command exists."""
    import shutil
    return shutil.which(cmd)


# ── Report persistence ────────────────────────────────────────────────


def save_research_report(
    stock_code: str,
    stock_name: str,
    report_text: str,
    skill: str,
    ai_berkshire_path: str = DEFAULT_AI_BERKSHIRE_PATH,
) -> Path:
    """Save research report to ai-berkshire reports directory."""
    reports_dir = Path(ai_berkshire_path) / "reports"
    stock_dir = reports_dir / stock_name
    stock_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"{stock_name}-dsa-signal-{date_str}.md"
    filepath = stock_dir / filename

    report_content = (
        f"# 深度研究：{stock_name} ({stock_code})\n\n"
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"**触发来源**: daily_stock_analysis 信号驱动\n"
        f"**研究技能**: ai-berkshire/{skill}\n"
        f"**数据截止**: {datetime.now().strftime('%Y-%m-%d')}\n\n"
        f"---\n\n"
        f"{report_text}\n"
    )

    filepath.write_text(report_content, encoding="utf-8")
    logger.info("Research report saved: %s", filepath)
    return filepath


# ── Main flow ─────────────────────────────────────────────────────────


def run_pipeline(
    stocks: List[Dict[str, Any]],
    skill: str = DEFAULT_SKILL,
    ai_berkshire_path: str = DEFAULT_AI_BERKSHIRE_PATH,
    claude_cli: str = DEFAULT_CLAUDE_CLI,
    dry_run: bool = False,
    feishu_webhook: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run the full bridge pipeline for a list of stocks.

    Returns list of research result dicts.
    """
    results = []

    for i, stock in enumerate(stocks):
        code = stock["code"]
        name = stock["name"]
        logger.info(
            "[%d/%d] Researching %s (%s) — score=%d signal=%s",
            i + 1, len(stocks), name, code,
            stock.get("score", 0),
            stock.get("signal_type", "unknown"),
        )

        question = resolve_question_for_stock(stock, skill)

        if dry_run:
            logger.info("[DRY RUN] Would research: %s (%s)", name, code)
            logger.info("[DRY RUN] Question: %s", question[:200])
            results.append({
                "stock_code": code,
                "stock_name": name,
                "status": "dry_run",
                "report_path": None,
            })
            continue

        # Invoke ai-berkshire research
        success, report = invoke_ai_berkshire_research(
            stock_code=code,
            stock_name=name,
            question=question,
            skill=skill,
            ai_berkshire_path=ai_berkshire_path,
            claude_cli=claude_cli,
        )

        if success:
            # Save report
            filepath = save_research_report(
                stock_code=code,
                stock_name=name,
                report_text=report,
                skill=skill,
                ai_berkshire_path=ai_berkshire_path,
            )

            result_entry = {
                "stock_code": code,
                "stock_name": name,
                "status": "success",
                "report_path": str(filepath),
                "report_preview": report[:500],
            }
            results.append(result_entry)

            # Send Feishu notification
            if feishu_webhook:
                title = f"🔬 深度研究：{name} ({code})"
                feishu_content = (
                    f"**股票**: {name} ({code})\n"
                    f"**评分**: {stock.get('score', 'N/A')}\n"
                    f"**信号**: {'📈 看多' if stock.get('signal_type') == 'bullish' else '📉 看空'}\n\n"
                    f"**研究摘要**:\n{report[:3000]}"
                )
                _send_feishu_notification(webhook_url, title, feishu_content)
        else:
            logger.error("Research failed for %s: %s", code, report[:200])
            results.append({
                "stock_code": code,
                "stock_name": name,
                "status": "failed",
                "error": report[:500],
            })

        # Rate limit: small delay between research calls
        if i < len(stocks) - 1:
            time.sleep(3)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ai-berkshire bridge — deep research for daily_stock_analysis signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Research a single stock\n"
            "  python scripts/ai_berkshire_bridge.py --stock 600519 --stock-name 贵州茅台\n\n"
            "  # Process analysis results from file\n"
            "  python scripts/ai_berkshire_bridge.py --results reports/latest.json\n\n"
            "  # Dry run to see which stocks would be researched\n"
            "  python scripts/ai_berkshire_bridge.py --results reports/latest.json --dry-run\n\n"
            "  # Custom minimum signal threshold\n"
            "  python scripts/ai_berkshire_bridge.py --results reports/latest.json --min-score 80\n"
        ),
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--stock", type=str,
        help="Single stock code to research",
    )
    input_group.add_argument(
        "--results", type=str,
        help="Path to analysis results JSON file or directory",
    )
    input_group.add_argument(
        "--stocks", type=str,
        help="Comma-separated list of stock codes (e.g., 600519,AAPL,hk00700)",
    )

    parser.add_argument("--stock-name", type=str, default="", help="Stock name (for single stock mode)")
    parser.add_argument("--question", type=str, default="", help="Custom research question")
    parser.add_argument(
        "--skill", type=str, default=os.environ.get("AI_BERKSHIRE_SKILL", DEFAULT_SKILL),
        help=f"ai-berkshire skill to use (default: {DEFAULT_SKILL})",
    )
    parser.add_argument(
        "--min-score", type=int, default=int(os.environ.get("BRIDGE_SIGNAL_MIN", str(DEFAULT_SIGNAL_MIN))),
        help=f"Min score to trigger research: >=min bullish, <=(100-min) bearish (default: {DEFAULT_SIGNAL_MIN})",
    )
    parser.add_argument(
        "--max-signals", type=int, default=int(os.environ.get("BRIDGE_MAX_SIGNALS", str(DEFAULT_MAX_SIGNALS))),
        help=f"Max stocks to research per run, most extreme first (default: {DEFAULT_MAX_SIGNALS}, 0=unlimited)",
    )
    parser.add_argument(
        "--ai-berkshire-path", type=str,
        default=os.environ.get("AI_BERKSHIRE_PATH", DEFAULT_AI_BERKSHIRE_PATH),
        help=f"Path to ai-berkshire repo (default: {DEFAULT_AI_BERKSHIRE_PATH})",
    )
    parser.add_argument(
        "--claude-cli", type=str,
        default=os.environ.get("CLAUDE_CLI_PATH", ""),
        help="Path to Claude CLI (default: auto-detect)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be researched without executing",
    )
    parser.add_argument(
        "--no-feishu", action="store_true",
        help="Skip Feishu notification",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve Claude CLI
    claude_cli = args.claude_cli or _find_claude_cli()
    logger.debug("Using Claude CLI: %s", claude_cli)

    # Resolve Feishu webhook
    feishu_webhook = None if args.no_feishu else _read_feishu_webhook()
    if feishu_webhook:
        logger.info("Feishu notification enabled")
    else:
        logger.info("Feishu notification disabled")

    # Build stock list
    stocks_to_research: List[Dict[str, Any]] = []

    if args.stock:
        # Single stock mode
        stock_name = args.stock_name or args.stock
        stocks_to_research.append({
            "code": args.stock,
            "name": stock_name,
            "score": 80,  # Assume strong signal for manual invocation
            "signal_type": "bullish",
            "action": "buy",
            "reason": "手动触发深度研究",
        })
        logger.info("Single stock mode: %s (%s)", args.stock, stock_name)

    elif args.results:
        # Results file mode — detect signals
        analysis_results = parse_analysis_results(args.results)
        if not analysis_results:
            logger.error("No analysis results found in: %s", args.results)
            sys.exit(1)
        logger.info("Parsed %d analysis results", len(analysis_results))

        stocks_to_research = detect_signal_stocks(analysis_results, args.min_score, args.max_signals)
        logger.info(
            "Detected %d stocks with strong signals (min score: %d)",
            len(stocks_to_research), args.min_score,
        )

    elif args.stocks:
        # Comma-separated stocks mode
        codes = [c.strip() for c in args.stocks.split(",") if c.strip()]
        for code in codes:
            stocks_to_research.append({
                "code": code,
                "name": code,
                "score": 80,
                "signal_type": "bullish",
                "action": "buy",
                "reason": "手动批量触发深度研究",
            })
        logger.info("Batch mode: %d stocks", len(stocks_to_research))

    else:
        parser.print_help()
        sys.exit(1)

    if not stocks_to_research:
        logger.info("No stocks with strong signals detected. Nothing to research.")
        return

    # Show research candidates
    logger.info("=== Research Candidates ===")
    for s in stocks_to_research:
        logger.info(
            "  %-12s %-20s score=%3d signal=%s",
            s["code"], s["name"], s.get("score", 0), s.get("signal_type", "?"),
        )

    # Send Feishu notification about starting research
    if feishu_webhook and not args.dry_run:
        summary = "\n".join(
            f"• {s['name']} ({s['code']}) — 评分:{s.get('score','?')} "
            f"{'📈' if s.get('signal_type')=='bullish' else '📉'}"
            for s in stocks_to_research
        )
        _send_feishu_simple_message(
            feishu_webhook,
            f"🔬 ai-berkshire 深度研究启动\n检测到 {len(stocks_to_research)} 只信号股票：\n{summary}",
        )

    # Run pipeline
    results = run_pipeline(
        stocks=stocks_to_research,
        skill=args.skill,
        ai_berkshire_path=args.ai_berkshire_path,
        claude_cli=claude_cli,
        dry_run=args.dry_run,
        feishu_webhook=feishu_webhook,
    )

    # Summary
    success_count = sum(1 for r in results if r["status"] == "success")
    fail_count = sum(1 for r in results if r["status"] == "failed")

    logger.info("=== Research Complete ===")
    logger.info("Success: %d, Failed: %d", success_count, fail_count)

    for r in results:
        if r["status"] == "success":
            logger.info("  ✅ %s (%s): %s", r["stock_name"], r["stock_code"], r["report_path"])
        elif r["status"] == "failed":
            logger.info("  ❌ %s (%s): %s", r["stock_name"], r["stock_code"], r.get("error", "Unknown")[:100])

    # Output results JSON to stdout for piping
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
