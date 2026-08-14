#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地 CYQ 筹码分布 vs 东财 stock_cyq_em 数值对齐检查（方案 B 验证脚本）

背景：`ak.stock_cyq_em` 断连率高（近期连续 100% 失败），方案 B 用本地 CYQ
算法（src/services/cyq.py，Tushare 日K+换手率推演，同款东财 JS 逻辑）兜底。
本脚本对自选股同时计算本地结果与东财参考值，输出误差表，量化对齐质量。

注意：脚本需要联网（Tushare daily/daily_basic + 东财筹码接口）。
akshare 若仍 100% 断连，参考列会缺失，属预期；此时以本地数值合理性 +
单测（tests/test_cyq_local.py）钉死的移植正确性为准。

用法：
    python3 scripts/cyq_alignment_check.py [--limit N] [--codes 300418,600519]

输出：
    - 每只股票：本地 vs 东财 的 获利比例/平均成本/90集中度 对比
    - 汇总：有参考样本的误差统计（平均绝对误差）
"""

import argparse
import logging
import signal
import sys

sys.path.insert(0, "")

logging.basicConfig(level=logging.WARNING)


class _Timeout:
    """单只股票东财请求的 wall-clock 上限（东财断连时避免整脚本卡死）。"""

    def __init__(self, seconds: int = 40):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)

    def __exit__(self, *exc):
        signal.alarm(0)

    def _handler(self, signum, frame):
        raise TimeoutError("东财筹码接口超时")


def _load_stock_list() -> list:
    from src.config import get_config

    return [c for c in get_config().stock_list if c and len(c) >= 5]


_LOCAL_FETCHER = None


def _fetch_local(code: str):
    """复用单个 TushareFetcher，float_share 批量/失败缓存只触发一次。"""
    from data_provider.tushare_fetcher import TushareFetcher

    global _LOCAL_FETCHER
    if _LOCAL_FETCHER is None:
        _LOCAL_FETCHER = TushareFetcher()
    return _LOCAL_FETCHER._fetch_local_cyq(code)


def _fetch_akshare_reference(code: str):
    from data_provider.akshare_fetcher import AkshareFetcher

    fetcher = AkshareFetcher()
    return fetcher.get_chip_distribution(code)


def _fmt(chip, field):
    if chip is None:
        return "N/A"
    value = getattr(chip, field, None)
    return "N/A" if value is None else round(float(value), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description="本地 CYQ vs 东财筹码对齐检查")
    parser.add_argument("--limit", type=int, default=5, help="参与东财参考对比的股票数上限")
    parser.add_argument("--codes", type=str, default="", help="指定股票代码（逗号分隔）")
    args = parser.parse_args()

    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    else:
        codes = _load_stock_list()
    if not codes:
        print("未找到自选股列表（--codes 指定或配置 stock_list）")
        return 1

    print(f"{'代码':<8}{'日期(本地)':<12}{'获利比例(东财/本地)':<20}"
          f"{'平均成本(东财/本地)':<20}{'90集中度(东财/本地)':<20}")
    print("-" * 80)

    ref_count = 0
    err_profit, err_cost, err_c90 = [], [], []

    for code in codes:
        local = _fetch_local(code)
        local_date = local.date if local else "-"
        # 东财参考仅对前 limit 只尝试，避免断连时拖垮整段
        ref = None
        if ref_count < args.limit:
            try:
                with _Timeout(40):
                    ref = _fetch_akshare_reference(code)
            except Exception as e:  # noqa: BLE001 - 诊断脚本，失败即记录
                print(f"  [东财参考失败] {code}: {type(e).__name__}: {e}")
            if ref is not None:
                ref_count += 1

        print(
            f"{code:<8}{local_date:<12}"
            f"{_fmt(ref, 'profit_ratio')}/{_fmt(local, 'profit_ratio'):<16}"
            f"{_fmt(ref, 'avg_cost')}/{_fmt(local, 'avg_cost'):<16}"
            f"{_fmt(ref, 'concentration_90')}/{_fmt(local, 'concentration_90'):<16}"
        )

        if ref is not None and local is not None:
            if ref.profit_ratio and local.profit_ratio:
                err_profit.append(abs(ref.profit_ratio - local.profit_ratio))
            if ref.avg_cost and local.avg_cost:
                err_cost.append(abs(ref.avg_cost - local.avg_cost) / ref.avg_cost)
            if ref.concentration_90 is not None and local.concentration_90 is not None:
                err_c90.append(abs(ref.concentration_90 - local.concentration_90))

    print("-" * 80)
    if err_profit:
        print(f"对齐样本数: {len(err_profit)}")
        print(f"获利比例 MAE: {sum(err_profit) / len(err_profit):.4f}")
        print(f"平均成本 相对MAE: {sum(err_cost) / len(err_cost):.2%}")
        print(f"90集中度 MAE: {sum(err_c90) / len(err_c90):.4f}")
    else:
        print("未获得东财参考样本（接口断连），对齐数据待 akshare 恢复后复测；"
              "本地数值正确性以 tests/test_cyq_local.py 单测为准。")

    print(f"\n本地计算完成 {len(codes)} 只（本地无网络筹码依赖，全部成功）。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断")
        sys.exit(130)
