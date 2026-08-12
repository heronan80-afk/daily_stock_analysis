"""从 reports/report_*.md 决策仪表盘中抽取个股信号。

报告每只股票一行的格式:
    <emoji> **名称(代码)**: 操作 | 评分 N | 看多/看空/震荡偏多/震荡偏空/震荡

示例:
    🟠 **中际旭创(300308)**: 减仓 | 评分 35 | 看空
    ⚪ **申菱环境(301018)**: 持有观察 | 评分 59 | 看多
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List, Optional


# 行尾方向词：看多/看空 有方向性；震荡偏多/震荡偏空 为弱方向性；震荡 为中性
_DIRECTION_MAP = {
    "看多": "long",
    "看空": "short",
    "震荡": "neutral",
    "震荡偏多": "long_lean",
    "震荡偏空": "short_lean",
}

# action 归一化到 long/short/hold
_ACTION_MAP = {
    "买入": "long",
    "加仓": "long",
    "卖出": "short",
    "减仓": "short",
    "持有观察": "hold",
    "观望": "hold",
    "持有": "hold",
}

# 匹配: **名称(代码)**: 操作 | 评分 N | 看多/看空
# 名称可含中文/字母/数字，代码 6 位数字，操作为中文短语，评分整数
_SIGNAL_RE = re.compile(
    r"\*\*(?P<name>[^(]+?)\((?P<code>\d{6})\)\*\*:\s*"
    r"(?P<action>[^|]+?)\s*\|\s*"
    r"评分\s*(?P<score>\d+)\s*\|\s*"
    r"(?P<direction>看多|看空|震荡偏多|震荡偏空|震荡)"
)

# 报告日期：# 🎯 2026-07-10 决策仪表盘
_DATE_RE = re.compile(r"^#\s*.+?(?P<date>\d{4}-\d{2}-\d{2})", re.MULTILINE)


@dataclass
class Signal:
    """单只股票在一份报告里的方向性信号。"""

    report_date: str  # 报告日期 YYYY-MM-DD
    code: str  # 6 位股票代码
    name: str  # 股票简称
    action: str  # 原始操作文本，如 "减仓" / "持有观察"
    action_kind: str  # 归一化: long / short / hold
    score: int  # 评分
    direction: str  # 看多 / 看空
    direction_kind: str  # long / short
    raw_line: str  # 原始行，便于排查

    @property
    def effective_direction(self) -> str:
        """实际判定用的方向。direction 优先，缺失时回退到 action_kind。"""
        if self.direction_kind in ("long", "short"):
            return self.direction_kind
        return self.action_kind

    def to_dict(self) -> dict:
        return asdict(self)


def parse_report(report_path: Path) -> List[Signal]:
    """解析单份报告，返回其中所有个股信号（含 hold）。

    报告是盘后生成，report_date 即为基准日 T。
    """
    text = report_path.read_text(encoding="utf-8")
    m = _DATE_RE.search(text)
    if not m:
        return []
    report_date = m.group("date")

    signals: List[Signal] = []
    for line in text.splitlines():
        m = _SIGNAL_RE.search(line)
        if not m:
            continue
        action_raw = m.group("action").strip()
        action_kind = _normalize_action(action_raw)
        direction = m.group("direction")
        sig = Signal(
            report_date=report_date,
            code=m.group("code"),
            name=m.group("name").strip(),
            action=action_raw,
            action_kind=action_kind,
            score=int(m.group("score")),
            direction=direction,
            direction_kind=_DIRECTION_MAP[direction],
            raw_line=line.strip(),
        )
        signals.append(sig)
    return signals


def _normalize_action(action_raw: str) -> str:
    """把原始操作文本归一化到 long/short/hold。"""
    for key, kind in _ACTION_MAP.items():
        if key in action_raw:
            return kind
    return "hold"


def iter_report_files(reports_dir: Path) -> Iterator[Path]:
    """遍历 reports/ 下的 report_*.md（不含 market_review_*）。"""
    yield from sorted(reports_dir.glob("report_*.md"))


def load_all_signals(reports_dir: Path) -> List[Signal]:
    """加载所有报告的信号，按 (report_date, code) 去重保序。"""
    seen = set()
    out: List[Signal] = []
    for path in iter_report_files(reports_dir):
        for sig in parse_report(path):
            key = (sig.report_date, sig.code)
            if key in seen:
                continue
            seen.add(key)
            out.append(sig)
    return out
