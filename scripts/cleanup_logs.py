#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理 daily_stock_analysis 的历史日志与报告。

为什么需要它：logging_config.py 用「按日期分文件 + RotatingFileHandler」，
但 RotatingFileHandler 只轮转当天文件，旧日期的 stock_analysis_debug_*.log[.N]
永不清理，会无限累积（单日可达 ~80MB）。本脚本按 mtime 清理 N 天前的历史文件。

安全设计：
- 默认 DRY-RUN，只列出将删的文件，不实际删除。
- --execute 才真删。
- 不碰 launchd_stdout/stderr.log（常驻进程正在写，截断有风险）。
- 不碰当天和近 N 天的文件。

用法：
  # 干跑看清单
  python scripts/cleanup_logs.py
  # 真删
  python scripts/cleanup_logs.py --execute
  # 自定义保留天数
  python scripts/cleanup_logs.py --log-keep-days 5 --report-keep-days 60 --execute
"""
import argparse
import time
from pathlib import Path


# 日志目录下要清理的文件名 glob（按 mtype 判断是否过期）
LOG_GLOBS = [
    "stock_analysis_debug_*.log*",   # 调试日志 + .1/.2 轮转备份（大头）
    "stock_analysis_2026*.log",       # 常规日志
    "manual_run_*.log",               # 手动补跑日志
    "run_*_stdout.log",               # 一次性运行日志
    "run_*_glm52.log",
    "tencent_manual_*.log",           # 腾讯报告手动发送日志
]

# reports 目录下要清理的文件名 glob
REPORT_GLOBS = [
    "report_2026*.md",
    "market_review_2026*.md",
]

# 绝不清理的文件（常驻进程在写，或当前活跃）
PROTECTED = {
    "launchd_stdout.log",
    "launchd_stderr.log",
    "berkshire_bridge_stdout.log",
    "berkshire_bridge_stderr.log",
}


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def collect_old_files(directory: Path, globs, keep_days: int):
    """返回 directory 下 mtime 早于 keep_days 天前、匹配 globs 的文件列表。"""
    cutoff = time.time() - keep_days * 86400
    files = []
    for pat in globs:
        for f in directory.glob(pat):
            if not f.is_file():
                continue
            if f.name in PROTECTED:
                continue
            if f.stat().st_mtime < cutoff:
                files.append(f)
    return sorted(set(files), key=lambda p: p.name)


def main():
    ap = argparse.ArgumentParser(description="清理 daily_stock_analysis 历史日志与报告")
    ap.add_argument("--root", default=None, help="项目根目录（默认脚本上级目录）")
    ap.add_argument("--log-keep-days", type=int, default=7, help="日志保留天数（默认 7）")
    ap.add_argument("--report-keep-days", type=int, default=30, help="报告保留天数（默认 30）")
    ap.add_argument("--execute", action="store_true", help="实际执行删除（默认 dry-run）")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    logs_dir = root / "logs"
    reports_dir = root / "reports"

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"=== {mode} | 日志保留 {args.log_keep_days} 天 / 报告保留 {args.report_keep_days} 天 ===")
    print(f"项目根: {root}")
    print()

    total = 0

    # 1. 旧日志
    old_logs = collect_old_files(logs_dir, LOG_GLOBS, args.log_keep_days) if logs_dir.is_dir() else []
    print(f"--- 旧日志 ({len(old_logs)} 个, {logs_dir.name}/) ---")
    for f in old_logs:
        s = f.stat().st_size
        total += s
        action = "删除" if args.execute else "将删"
        print(f"  {action} {human_size(s):>9}  {f.name}")
        if args.execute:
            f.unlink()
    if not old_logs:
        print("  （无过期日志）")
    print()

    # 2. 旧报告
    old_reports = collect_old_files(reports_dir, REPORT_GLOBS, args.report_keep_days) if reports_dir.is_dir() else []
    print(f"--- 旧报告 ({len(old_reports)} 个, {reports_dir.name}/) ---")
    for f in old_reports:
        s = f.stat().st_size
        total += s
        action = "删除" if args.execute else "将删"
        print(f"  {action} {human_size(s):>9}  {f.name}")
        if args.execute:
            f.unlink()
    if not old_reports:
        print("  （无过期报告）")
    print()

    print(f"==> {'已释放' if args.execute else '可释放'}: {human_size(total)}")
    if not args.execute and total > 0:
        print("（dry-run，未实际删除。加 --execute 真删。）")


if __name__ == "__main__":
    main()
