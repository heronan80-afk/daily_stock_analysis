# -*- coding: utf-8 -*-
"""BaostockFetcher.get_chip_distribution（本地筹码后备源）测试。

用 mock 的 baostock 会话验证：真实换手率 turn 驱动本地 CYQ 分布、空数据/退化/
查询失败/不支持市场时的 None 返回。全程不触网。
"""

import unittest
from unittest.mock import patch

from data_provider.baostock_fetcher import BaostockFetcher


class _FakeLogin:
    def __init__(self, code, msg=""):
        self.error_code = code
        self.error_msg = msg


class _FakeRs:
    """模拟 baostock query_history_k_data_plus 的返回对象（error_code + next/行迭代）。"""

    def __init__(self, fields, rows, error_code="0", error_msg=""):
        self.error_code = error_code
        self.error_msg = error_msg
        self.fields = fields
        self._rows = rows
        self._i = 0

    def next(self):
        if self._i < len(self._rows):
            return True
        return False

    def get_row_data(self):
        row = self._rows[self._i]
        self._i += 1
        return row


class _FakeBs:
    """模拟 baostock 模块：login/logout 成功 + 固定日 K 返回。"""

    _FIELDS = ["date", "open", "high", "low", "close", "volume", "turn"]

    def __init__(self, rows):
        self._rows = rows
        self._queried = []

    def login(self):
        return _FakeLogin("0", "success")

    def logout(self):
        return _FakeLogin("0", "success")

    def query_history_k_data_plus(self, **kwargs):
        self._queried.append(kwargs)
        return _FakeRs(self._FIELDS, self._rows)


def _fake_bs(rows):
    return _FakeBs(rows)


class TestBaostockChipBackup(unittest.TestCase):
    """Baostock 本地筹码后备源：真实 turn 换手率驱动 calc_cyq_metrics。"""

    def _make_fetcher(self, rows):
        """构造 fetcher + fake baostock（不 patch，调用处自行包 patch 上下文）。"""
        fetcher = BaostockFetcher()
        bs = _fake_bs(rows)
        return fetcher, bs

    def test_chip_from_real_turn_turnover(self) -> None:
        """真实 turn（百分数）驱动分布：语义与 cyq 两日衰减用例一致，
        source 标记 baostock_local，日期取最后一个交易日。"""
        rows = [
            ["2026-07-14", "10.0", "10.0", "10.0", "10.0", "1000", "100.0"],
            ["2026-07-15", "20.0", "20.0", "20.0", "20.0", "2000", "50.0"],
        ]
        fetcher, bs = self._make_fetcher(rows)
        with patch.object(BaostockFetcher, "_get_baostock", return_value=bs):
            chip = fetcher.get_chip_distribution("600519")

        self.assertIsNotNone(chip)
        self.assertEqual(chip.source, "baostock_local")
        self.assertEqual(chip.date, "2026-07-15")
        # 第 1 天满换手 10 元，第 2 天 50% 换手 20 元 → 中位数成本在 20 元档
        self.assertEqual(chip.avg_cost, 20.0)
        self.assertAlmostEqual(chip.profit_ratio, 1.0)
        # 查询用了不复权口径（adjustflag=3）且包含 turn
        self.assertEqual(bs._queried[0]["adjustflag"], "3")
        self.assertIn("turn", bs._queried[0]["fields"])

    def test_none_when_no_rows(self) -> None:
        """日 K 为空 → None，不抛异常。"""
        fetcher, bs = self._make_fetcher([])
        with patch.object(BaostockFetcher, "_get_baostock", return_value=bs):
            self.assertIsNone(fetcher.get_chip_distribution("600519"))

    def test_none_when_turnover_all_zero(self) -> None:
        """窗口内换手全为 0 → 分布退化 → None。"""
        rows = [
            ["2026-07-14", "10.0", "10.5", "9.5", "10.2", "1000", "0.0"],
        ]
        fetcher, bs = self._make_fetcher(rows)
        with patch.object(BaostockFetcher, "_get_baostock", return_value=bs):
            self.assertIsNone(fetcher.get_chip_distribution("600519"))

    def test_none_when_query_error(self) -> None:
        """baostock 查询返回错误 → None，不抛异常。"""
        bs = _FakeBs([])

        def _err_query(**kwargs):
            return _FakeRs([], [], error_code="-1", error_msg="query failed")

        bs.query_history_k_data_plus = _err_query
        fetcher = BaostockFetcher()
        with patch.object(BaostockFetcher, "_get_baostock", return_value=bs):
            self.assertIsNone(fetcher.get_chip_distribution("600519"))

    def test_none_for_unsupported_market(self) -> None:
        """美股/港股/北交所直接返回 None，不发起 baostock 查询。"""
        for code in ("AAPL", "hk00700", "920001"):
            fetcher = BaostockFetcher()
            with patch.object(BaostockFetcher, "_get_baostock", return_value=_FakeBs([])):
                self.assertIsNone(fetcher.get_chip_distribution(code))

    def test_none_when_login_fails(self) -> None:
        """登录失败（error_code != 0）→ 抛 DataFetchError 被吞 → None。"""
        class _FailBs(_FakeBs):
            def login(self):
                return _FakeLogin("-1", "login failed")

        fetcher = BaostockFetcher()
        with patch.object(BaostockFetcher, "_get_baostock", return_value=_FailBs([])):
            self.assertIsNone(fetcher.get_chip_distribution("600519"))


if __name__ == "__main__":
    unittest.main()
