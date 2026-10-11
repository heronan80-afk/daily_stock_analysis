# -*- coding: utf-8 -*-
"""Unit tests for MxDataFetcher's index response parsing.

The fixtures below are real (trimmed) payloads captured from the miaoxiang
``/finskillshub/api/claw/query`` endpoint. The endpoint returns the same
information in several different table shapes depending on the wording of
``toolQuery``, and each shape used to be a silent data-loss trap:

- shape "columns are index names" (one indicator, many indices)
- shape "headName rows are index names" (many indices, many indicators)
- shape "single index" (title carries the index name, headName rows are dates)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.mxdata_fetcher import (
    MxDataFetcher,
    _parse_amount,
    _to_point,
)


def _dto(title, name_map, table):
    return {"title": title, "nameMap": name_map, "table": table}


# 形态：单指数（title 带指数名，headName 行是日期，列是指标编码）
SINGLE_INDEX_DTO = _dto(
    "上证指数(000001.SH)(指数)的成交量、成交额等",
    {
        "324785": "成交量",
        "headNameSub": "数据来源",
        "327483": "成交额",
        "326386": "最低价",
        "325898": "收盘价",
        "326339": "最高价",
        "326865": "涨跌幅",
    },
    {
        "324785": ["535.1亿股", "482.2亿股"],
        "327483": ["8886亿", "8112亿"],
        "326386": ["3754.1353点", "3795.374点"],
        "325898": ["3813.7913点", "3811.9043点"],
        "326339": ["3824.7091点", "3864.1181点"],
        "326865": ["0.0495", "-0.7884"],
        "headName": ["2026-10-09(日)", "2026-10-08(日)"],
    },
)

# 形态：headName 行是指数名（一次查询多指数多指标）—— 曾整段被丢弃
ROWS_ARE_INDICES_DTO = _dto(
    "上证指数(000001.SH)(指数)等当前的最低价、最新价等",
    {
        "ZDJ_f16_3": "最低价",
        "headNameSub": "数据来源",
        "ZXJ_f2_3": "最新价",
        "ZGJ_f15_3": "最高价",
        "ZF_f7_3": "振幅",
        "ZDF_f3_3": "涨跌幅",
        "KPJ_f17_3": "开盘价",
        "CJE_f6_3": "成交额",
        "CJL_f5_3": "成交量",
        "securityName": "股票名称",
    },
    {
        "ZDJ_f16_3": ["1.228万", "2933", "1390"],
        "ZXJ_f2_3": ["12641.86", "3043.33", "1457.27"],
        "ZGJ_f15_3": ["1.268万", "3053", "1467"],
        "ZF_f7_3": ["3.18%", "3.97%", "5.28%"],
        "ZDF_f3_3": ["0.17%", "0.22%", "0.07%"],
        "KPJ_f17_3": ["12557.02", "3021.33", "1441.72"],
        "CJE_f6_3": ["1.012万亿", "4948亿", "919.7亿"],
        "CJL_f5_3": ["632.7亿", "197.1亿", "9.541亿"],
        "headName": [
            "深证成指(399001.SZ)(指数)",
            "创业板指(399006.SZ)(指数)",
            "科创50(000688.SH)(指数)",
        ],
    },
)

# 形态：列名是指数名（单一指标，多指数）
COLUMNS_ARE_INDICES_DTO = _dto(
    "上证指数(000001.SH)(指数)等的振幅",
    {"100000000020903": "振幅", "headNameSub": "数据来源"},
    {
        "上证指数(000001.SH)(指数)": ["1.851"],
        "深证成指(399001.SZ)(指数)": ["3.176"],
        "创业板指(399006.SZ)(指数)": ["3.971"],
        "科创50(000688.SH)(指数)": ["5.279"],
        "上证50(000016.SH)(指数)": ["1.601"],
        "headName": ["2026-10-09"],
    },
)


class MxDataFetcherParseTestCase(unittest.TestCase):
    def setUp(self):
        self.fetcher = MxDataFetcher()

    def _parse(self, *dtos):
        out = {}
        for dto in dtos:
            self.fetcher._parse_dto(dto, out)
        return out

    def test_single_index_shape_parses_latest_row(self):
        out = self._parse(SINGLE_INDEX_DTO)
        self.assertEqual(set(out), {"上证指数"})
        self.assertEqual(out["上证指数"]["current"], "3813.7913点")  # 取 headName 第 0 行（最新）
        self.assertEqual(out["上证指数"]["change_pct"], "0.0495")
        self.assertEqual(out["上证指数"]["high"], "3824.7091点")
        self.assertEqual(out["上证指数"]["low"], "3754.1353点")
        self.assertEqual(out["上证指数"]["amount"], "8886亿")

    def test_rows_are_indices_shape_keeps_every_index(self):
        """Regression: the 2nd/3rd index of such a DTO used to be dropped."""
        out = self._parse(ROWS_ARE_INDICES_DTO)
        self.assertEqual(set(out), {"深证成指", "创业板指", "科创50"})
        self.assertEqual(out["深证成指"]["current"], "12641.86")
        self.assertEqual(out["创业板指"]["current"], "3043.33")
        self.assertEqual(out["科创50"]["current"], "1457.27")

    def test_rows_are_indices_shape_parses_open_and_amplitude(self):
        out = self._parse(ROWS_ARE_INDICES_DTO)
        self.assertEqual(out["深证成指"]["open"], "12557.02")
        self.assertEqual(out["创业板指"]["open"], "3021.33")
        self.assertEqual(out["科创50"]["open"], "1441.72")
        self.assertEqual(out["创业板指"]["amplitude"], "3.97%")

    def test_columns_are_indices_shape_parses_single_indicator(self):
        out = self._parse(COLUMNS_ARE_INDICES_DTO)
        self.assertEqual(len(out), 5)
        self.assertEqual(out["上证指数"]["amplitude"], "1.851")
        self.assertEqual(out["科创50"]["amplitude"], "5.279")

    def test_shapes_combine_without_losing_fields(self):
        out = self._parse(
            SINGLE_INDEX_DTO,
            ROWS_ARE_INDICES_DTO,
            COLUMNS_ARE_INDICES_DTO,
        )
        # 后解析的 DTO 覆盖同名字段，但每个指数最终都该拿到 open
        for name in ("深证成指", "创业板指", "科创50"):
            self.assertIn("open", out[name])
        self.assertEqual(out["科创50"]["open"], "1441.72")

    def test_build_index_dict_converts_values(self):
        out = self._parse(ROWS_ARE_INDICES_DTO)
        row = MxDataFetcher._build_index_dict("深证成指", "sz399001", out["深证成指"])
        self.assertEqual(row["code"], "sz399001")
        self.assertAlmostEqual(row["current"], 12641.86, places=2)
        self.assertAlmostEqual(row["change_pct"], 0.17, places=2)
        self.assertAlmostEqual(row["open"], 12557.02, places=2)
        self.assertAlmostEqual(row["low"], 12280.0, places=2)  # "1.228万"
        self.assertAlmostEqual(row["amplitude"], 3.18, places=2)
        self.assertAlmostEqual(row["amount"], 1.012e12)

    def test_unrelated_tables_are_ignored(self):
        """Constituent-stock tables must not be mistaken for index quotes."""
        junk = _dto(
            "None",
            {"a": "证券简称", "b": "收盘价", "c": "申万一级行业"},
            {
                "a": ["平安银行"],
                "b": ["11.2"],
                "c": ["银行"],
                "headName": ["证券简称", "收盘价", "涨跌"],
            },
        )
        self.assertEqual(self._parse(junk), {})


class MxDataFetcherNumericTestCase(unittest.TestCase):
    def test_to_point_handles_chinese_units(self):
        self.assertAlmostEqual(_to_point("3804.0896"), 3804.0896, places=4)
        self.assertAlmostEqual(_to_point("3804.09点"), 3804.09, places=4)
        self.assertAlmostEqual(_to_point("1.256万"), 12560.0, places=2)
        self.assertAlmostEqual(_to_point("1.228万"), 12280.0, places=2)
        self.assertAlmostEqual(_to_point("1.5亿"), 1.5e8)

    def test_to_point_empty_values(self):
        for value in (None, "", "-", "None", "nan"):
            self.assertEqual(_to_point(value), 0.0)


class MxDataFetcherAvailabilityTestCase(unittest.TestCase):
    def test_only_serves_main_indices(self):
        fetcher = MxDataFetcher()
        fetcher._api_key = "dummy"
        self.assertTrue(fetcher.is_available_for_request("main_indices"))
        for capability in ("daily_data", "realtime_quote", "stock_name", "stock_list"):
            self.assertFalse(fetcher.is_available_for_request(capability))

    def test_unavailable_without_api_key(self):
        fetcher = MxDataFetcher()
        fetcher._api_key = ""
        self.assertFalse(fetcher.is_available_for_request("main_indices"))


if __name__ == "__main__":
    unittest.main()
