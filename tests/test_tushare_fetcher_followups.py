# -*- coding: utf-8 -*-
"""Regression tests for post-merge Tushare follow-up fixes."""

import importlib.util
import os
import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

try:
    json_repair_available = importlib.util.find_spec("json_repair") is not None
except ValueError:
    json_repair_available = "json_repair" in sys.modules

if not json_repair_available and "json_repair" not in sys.modules:
    sys.modules["json_repair"] = MagicMock()

from data_provider.tushare_fetcher import TushareFetcher
from data_provider.realtime_types import ChipDistribution


class TestTushareFetcherFollowUps(unittest.TestCase):
    """Cover rate limiting and cross-day trade-calendar refresh behavior."""

    def setUp(self) -> None:
        # 本地筹码 float_share 磁盘缓存指向临时路径，避免污染真实 data/cache/
        self._tmp_float_cache = os.path.join(tempfile.mkdtemp(), "float_share.json")

    @staticmethod
    def _make_fetcher() -> TushareFetcher:
        with patch.object(TushareFetcher, "_init_api", return_value=None):
            fetcher = TushareFetcher()
        fetcher._api = MagicMock()
        fetcher.priority = 2
        return fetcher

    def test_get_trade_time_refreshes_trade_calendar_when_day_changes(self) -> None:
        fetcher = self._make_fetcher()
        fetcher._api.trade_cal.side_effect = [
            pd.DataFrame({"cal_date": ["20260317", "20260314"], "is_open": [1, 1]}),
            pd.DataFrame({"cal_date": ["20260318", "20260317"], "is_open": [1, 1]}),
        ]

        with patch.object(
            fetcher,
            "_get_china_now",
            side_effect=[
                datetime(2026, 3, 17, 20, 0),
                datetime(2026, 3, 17, 20, 0),
                datetime(2026, 3, 18, 20, 0),
                datetime(2026, 3, 18, 20, 0),
            ],
        ), patch.object(fetcher, "_check_rate_limit") as rate_limit_mock:
            self.assertEqual(fetcher.get_trade_time(early_time="00:00", late_time="19:00"), "20260317")
            self.assertEqual(fetcher.get_trade_time(early_time="00:00", late_time="19:00"), "20260318")

        self.assertEqual(fetcher._api.trade_cal.call_count, 2)
        self.assertEqual(rate_limit_mock.call_count, 2)
    def test_get_trade_time_returns_latest_trade_date_on_non_trade_day(self) -> None:
        """Non-trade day (e.g. Saturday) should return the most recent trade
        date (Friday), not the one before it (Thursday).  Fixes #1009."""
        fetcher = self._make_fetcher()
        # 2026-03-21 is Saturday; Friday 20 and Thursday 19 are trade dates
        fetcher._api.trade_cal.return_value = pd.DataFrame(
            {
                "cal_date": ["20260314", "20260315", "20260316",
                             "20260317", "20260318", "20260319",
                             "20260320", "20260321"],
                "is_open": [0, 0, 1, 1, 1, 1, 1, 0],
            }
        )

        with patch.object(
            fetcher,
            "_get_china_now",
            # called twice: once by get_trade_time, once by _get_trade_dates
            side_effect=[datetime(2026, 3, 21, 10, 0)] * 2,
        ), patch.object(fetcher, "_check_rate_limit"):
            result = fetcher.get_trade_time(early_time="00:00", late_time="19:00")

        # Should be Friday (20th), NOT Thursday (19th)
        self.assertEqual(result, "20260320")

    def test_get_trade_time_trade_day_before_data_ready_returns_previous(self) -> None:
        """On a trade day within the early-late window, should return the
        previous trade date (data not ready yet for today)."""
        fetcher = self._make_fetcher()
        fetcher._api.trade_cal.return_value = pd.DataFrame(
            {
                "cal_date": ["20260319", "20260320"],
                "is_open": [1, 1],
            }
        )

        with patch.object(
            fetcher,
            "_get_china_now",
            # Friday 10:00 AM - within 00:00~19:00 window, data not ready
            side_effect=[datetime(2026, 3, 20, 10, 0)] * 2,
        ), patch.object(fetcher, "_check_rate_limit"):
            result = fetcher.get_trade_time(early_time="00:00", late_time="19:00")

        # Data not ready, should fall back to Thursday (19th)
        self.assertEqual(result, "20260319")
        
          
    def test_get_sector_rankings_rate_limits_calendar_and_rankings_api(self) -> None:
        fetcher = self._make_fetcher()
        fetcher._api.trade_cal.return_value = pd.DataFrame(
            {"cal_date": ["20260317", "20260314"], "is_open": [1, 1]}
        )
        fetcher._api.moneyflow_ind_ths.return_value = pd.DataFrame(
            {
                "industry": ["AI", "消费"],
                "pct_change": [1.8, -0.6],
            }
        )

        with patch.object(fetcher, "_get_china_now", return_value=datetime(2026, 3, 17, 16, 0)), patch.object(
            fetcher, "_check_rate_limit"
        ) as rate_limit_mock:
            top, bottom = fetcher.get_sector_rankings(n=1)

        self.assertEqual(top, [{"name": "AI", "change_pct": 1.8}])
        self.assertEqual(bottom, [{"name": "消费", "change_pct": -0.6}])
        self.assertEqual(rate_limit_mock.call_count, 2)

    def test_get_chip_distribution_rate_limits_all_tushare_calls(self) -> None:
        fetcher = self._make_fetcher()
        fetcher._api.trade_cal.return_value = pd.DataFrame(
            {"cal_date": ["20260317", "20260314"], "is_open": [1, 1]}
        )
        fetcher._api.cyq_chips.return_value = pd.DataFrame(
            {
                "price": [9.0, 10.0, 11.0],
                "percent": [20.0, 50.0, 30.0],
            }
        )
        fetcher._api.daily.return_value = pd.DataFrame({"close": [10.5]})

        with patch.object(fetcher, "_get_china_now", return_value=datetime(2026, 3, 17, 20, 0)), patch.object(
            fetcher, "_check_rate_limit"
        ) as rate_limit_mock:
            chip = fetcher.get_chip_distribution("600519")

        self.assertIsNotNone(chip)
        if chip is None:
            self.fail("expected chip distribution data")
        self.assertEqual(chip.date, "2026-03-17")
        self.assertAlmostEqual(chip.profit_ratio, 0.7)
        self.assertAlmostEqual(chip.avg_cost, 10.1)
        self.assertAlmostEqual(chip.concentration_90, 0.1)
        self.assertAlmostEqual(chip.concentration_70, 0.1)
        self.assertEqual(rate_limit_mock.call_count, 3)

    def test_convert_stock_code_accepts_exchange_prefixed_a_share(self) -> None:
        fetcher = self._make_fetcher()

        self.assertEqual(fetcher._convert_stock_code("SZ000001"), "000001.SZ")
        self.assertEqual(fetcher._convert_stock_code("SH600519"), "600519.SH")
        self.assertEqual(fetcher._convert_stock_code("605218"), "605218.SH")
        self.assertEqual(fetcher._convert_stock_code("600519.SS"), "600519.SH")

    @patch.dict(sys.modules, {"tushare": MagicMock()})
    def test_legacy_realtime_quote_keeps_sz_hint_as_stock_symbol(self) -> None:
        fetcher = self._make_fetcher()
        fetcher._api.quotation.side_effect = Exception("quota")

        tushare_module = sys.modules["tushare"]
        tushare_module.get_realtime_quotes.return_value = pd.DataFrame(
            [
                {
                    "name": "平安银行",
                    "price": "10.94",
                    "pre_close": "10.88",
                    "volume": "1000",
                    "amount": "2000",
                    "high": "11.00",
                    "low": "10.80",
                    "open": "10.90",
                }
            ]
        )

        quote = fetcher.get_realtime_quote("SZ000001")

        self.assertIsNotNone(quote)
        self.assertEqual(quote.code, "000001")
        self.assertEqual(quote.name, "平安银行")
        tushare_module.get_realtime_quotes.assert_called_once_with("000001")

    def test_get_trade_dates_falls_back_to_cache_on_rate_limit(self) -> None:
        """When trade_cal is rate-limited, _get_trade_dates should return
        the cached trade dates instead of propagating the exception."""
        fetcher = self._make_fetcher()
        # Seed the cache so it can be reused when trade_cal fails
        fetcher.date_list = ["20260317", "20260314"]
        fetcher._date_list_end = "20260317"
        fetcher._api.trade_cal.side_effect = Exception(
            "抱歉，您访问接口(trade_cal)频率超限(1次/分钟)"
        )

        with patch.object(fetcher, "_get_china_now", return_value=datetime(2026, 3, 18, 20, 0)):
            result = fetcher._get_trade_dates("20260318")

        # Should return cached dates, not raise
        self.assertEqual(result, ["20260317", "20260314"])

    def test_get_trade_dates_falls_back_to_weekday_on_rate_limit_no_cache(self) -> None:
        """When trade_cal is rate-limited and no cache exists, _get_trade_dates
        should fall back to weekday-based dates instead of raising."""
        fetcher = self._make_fetcher()
        fetcher._api.trade_cal.side_effect = Exception(
            "抱歉，您访问接口(trade_cal)频率超限(1次/分钟)"
        )

        # 2026-03-20 is Friday
        with patch.object(fetcher, "_get_china_now", return_value=datetime(2026, 3, 20, 20, 0)):
            result = fetcher._get_trade_dates("20260320")

        # Should not raise; should contain weekday dates (skip Sat/Sun)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("20260320", result)  # Friday
        self.assertNotIn("20260321", result)  # Saturday
        self.assertNotIn("20260322", result)  # Sunday

    def test_get_chip_distribution_survives_trade_cal_rate_limit(self) -> None:
        """get_chip_distribution should not crash when trade_cal is rate-limited
        but cached trade dates are available."""
        fetcher = self._make_fetcher()
        # Seed the cache
        fetcher.date_list = ["20260317", "20260314"]
        fetcher._date_list_end = "20260317"
        # trade_cal fails, but cyq_chips and daily succeed
        fetcher._api.trade_cal.side_effect = Exception(
            "抱歉，您访问接口(trade_cal)频率超限(1次/分钟)"
        )
        fetcher._api.cyq_chips.return_value = pd.DataFrame(
            {"price": [9.0, 10.0, 11.0], "percent": [20.0, 50.0, 30.0]}
        )
        fetcher._api.daily.return_value = pd.DataFrame({"close": [10.5]})

        with patch.object(fetcher, "_get_china_now", return_value=datetime(2026, 3, 18, 20, 0)):
            chip = fetcher.get_chip_distribution("600519")

        self.assertIsNotNone(chip)
        if chip is not None:
            self.assertEqual(chip.date, "2026-03-17")

    def test_get_chip_distribution_falls_back_to_local_cyq_when_no_permission(self) -> None:
        """cyq_chips 无接口权限时，本地 CYQ 算法（Tushare 日K+换手率）兜底返回分布。

        兜底后仍输出与 ChipDistribution 契约一致的字段，source 标记为 tushare_local，
        日期格式化为 YYYY-MM-DD。换手率走真实 float_share（daily_basic 批量）路径。
        """
        fetcher = self._make_fetcher()
        fetcher._api.trade_cal.return_value = pd.DataFrame(
            {"cal_date": ["20260814", "20260813"], "is_open": [1, 1]}
        )
        # cyq_chips 无权限 → 触发本地算法兜底
        fetcher._api.cyq_chips.side_effect = Exception(
            "抱歉，您没有接口(cyq_chips)访问权限"
        )

        # 合成 30 个交易日日 K（价格小幅波动）
        dates = pd.bdate_range("2026-07-01", periods=30).strftime("%Y%m%d").tolist()
        daily_rows = []
        price = 10.0
        for i, d in enumerate(dates):
            o = price
            c = price * (1.0 + (0.01 if i % 7 == 6 else -0.002))
            h = max(o, c) * 1.01
            l = min(o, c) * 0.99
            price = c
            daily_rows.append(
                {
                    "ts_code": "600519.SH",
                    "trade_date": d,
                    "open": round(o, 2),
                    "high": round(h, 2),
                    "low": round(l, 2),
                    "close": round(c, 2),
                    "vol": 50000.0,
                    "amount": 5e6,
                }
            )
        fetcher._api.daily.return_value = pd.DataFrame(daily_rows)
        # daily_basic 批量口径：trade_date 一次返回全市场 {ts_code, float_share}
        fetcher._api.daily_basic.return_value = pd.DataFrame(
            {"ts_code": ["600519.SH"], "float_share": [100000.0]}
        )

        with patch.object(TushareFetcher, "_FLOAT_SHARE_CACHE_FILE", self._tmp_float_cache), \
             patch.object(fetcher, "_get_china_now", return_value=datetime(2026, 8, 14, 18, 0)):
            chip = fetcher.get_chip_distribution("600519")

        self.assertIsNotNone(chip)
        if chip is not None:
            self.assertEqual(chip.source, "tushare_local")
            expected_date = datetime.strptime(dates[-1], "%Y%m%d").strftime("%Y-%m-%d")
            self.assertEqual(chip.date, expected_date)
            self.assertGreater(chip.avg_cost, 0)
            self.assertGreaterEqual(chip.concentration_90, 0)
            self.assertLessEqual(chip.concentration_90, 1)
            self.assertGreaterEqual(chip.profit_ratio, 0)
            self.assertLessEqual(chip.profit_ratio, 1)
            self.assertGreater(chip.cost_90_high, chip.cost_90_low)

    def test_get_chip_distribution_local_cyq_volume_proxy_when_no_float_share(self) -> None:
        """float_share 不可用（daily_basic 限流/失败）且 baostock 后备源也不可用时，
        退化为成交量代理换手率，仍能产出分布而不是缺失。"""
        fetcher = self._make_fetcher()
        fetcher._api.trade_cal.return_value = pd.DataFrame(
            {"cal_date": ["20260814", "20260813"], "is_open": [1, 1]}
        )
        fetcher._api.cyq_chips.side_effect = Exception(
            "抱歉，您没有接口(cyq_chips)访问权限"
        )
        dates = pd.bdate_range("2026-07-01", periods=30).strftime("%Y%m%d").tolist()
        daily_rows = []
        price = 10.0
        for i, d in enumerate(dates):
            o = price
            c = price * (1.0 + (0.01 if i % 7 == 6 else -0.002))
            h = max(o, c) * 1.01
            l = min(o, c) * 0.99
            price = c
            daily_rows.append(
                {
                    "ts_code": "600519.SH",
                    "trade_date": d,
                    "open": round(o, 2),
                    "high": round(h, 2),
                    "low": round(l, 2),
                    "close": round(c, 2),
                    "vol": 50000.0,
                    "amount": 5e6,
                }
            )
        fetcher._api.daily.return_value = pd.DataFrame(daily_rows)
        # daily_basic 批量拉取失败（限流）→ 走成交量代理
        fetcher._api.daily_basic.side_effect = Exception(
            "抱歉，您访问接口(daily_basic)频率超限(1次/分钟)"
        )

        # baostock 后备源不可用（如登录失败）→ 走成交量代理
        with patch.object(TushareFetcher, "_FLOAT_SHARE_CACHE_FILE", self._tmp_float_cache), \
             patch("data_provider.tushare_fetcher._baostock_local_chip", return_value=None), \
             patch.object(fetcher, "_get_china_now", return_value=datetime(2026, 8, 14, 18, 0)):
            chip = fetcher.get_chip_distribution("600519")

        self.assertIsNotNone(chip)
        if chip is not None:
            self.assertEqual(chip.source, "tushare_local")
            self.assertGreater(chip.avg_cost, 0)
            self.assertGreaterEqual(chip.concentration_90, 0)

    def test_get_chip_distribution_local_cyq_prefers_baostock_real_turnover(self) -> None:
        """无 float_share 时优先用 baostock 真实换手率源（source=baostock_local），
        而不是成交量代理；只有 baostock 不可用才退化到代理。"""
        fetcher = self._make_fetcher()
        fetcher._api.trade_cal.return_value = pd.DataFrame(
            {"cal_date": ["20260814", "20260813"], "is_open": [1, 1]}
        )
        fetcher._api.cyq_chips.side_effect = Exception(
            "抱歉，您没有接口(cyq_chips)访问权限"
        )
        dates = pd.bdate_range("2026-07-01", periods=30).strftime("%Y%m%d").tolist()
        daily_rows = []
        price = 10.0
        for i, d in enumerate(dates):
            o = price
            c = price * (1.0 + (0.01 if i % 7 == 6 else -0.002))
            h = max(o, c) * 1.01
            lo = min(o, c) * 0.99
            price = c
            daily_rows.append(
                {
                    "ts_code": "600519.SH",
                    "trade_date": d,
                    "open": round(o, 2),
                    "high": round(h, 2),
                    "low": round(lo, 2),
                    "close": round(c, 2),
                    "vol": 50000.0,
                    "amount": 5e6,
                }
            )
        fetcher._api.daily.return_value = pd.DataFrame(daily_rows)
        # daily_basic 批量拉取失败（限流）→ 走 baostock 后备源
        fetcher._api.daily_basic.side_effect = Exception(
            "抱歉，您访问接口(daily_basic)频率超限(1次/分钟)"
        )

        baostock_chip = ChipDistribution(
            code="600519",
            date="2026-08-14",
            source="baostock_local",
            profit_ratio=0.5,
            avg_cost=42.0,
            cost_90_low=30.0,
            cost_90_high=50.0,
            concentration_90=0.2,
            cost_70_low=35.0,
            cost_70_high=46.0,
            concentration_70=0.1,
        )
        with patch.object(TushareFetcher, "_FLOAT_SHARE_CACHE_FILE", self._tmp_float_cache), \
             patch("data_provider.tushare_fetcher._baostock_local_chip",
                   return_value=baostock_chip), \
             patch.object(fetcher, "_get_china_now", return_value=datetime(2026, 8, 14, 18, 0)):
            chip = fetcher.get_chip_distribution("600519")

        self.assertIsNotNone(chip)
        if chip is not None:
            self.assertEqual(chip.source, "baostock_local")
            self.assertEqual(chip.avg_cost, 42.0)

    def test_get_chip_distribution_local_cyq_returns_none_when_daily_missing(self) -> None:
        """本地兜底缺日 K 数据（或接口失败）时返回 None，不抛异常。"""
        fetcher = self._make_fetcher()
        fetcher._api.trade_cal.return_value = pd.DataFrame(
            {"cal_date": ["20260814", "20260813"], "is_open": [1, 1]}
        )
        fetcher._api.cyq_chips.side_effect = Exception(
            "抱歉，您没有接口(cyq_chips)访问权限"
        )
        fetcher._api.daily.return_value = pd.DataFrame()  # 空日 K

        with patch.object(fetcher, "_get_china_now", return_value=datetime(2026, 8, 14, 18, 0)):
            chip = fetcher.get_chip_distribution("600519")

        self.assertIsNone(chip)
