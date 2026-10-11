# -*- coding: utf-8 -*-
"""
===================================
MxDataFetcher - 妙想金融数据（优先指数数据源）
===================================

数据来源：东方财富妙想金融数据 API（mkapi2.dfcfs.com）
特点：官方权威行情/财务数据，自然语言查询，每日调用次数有限额（默认 150 次/天）
用途：仅作为**主要指数行情**的优先数据源；不支持日线/实时个股行情。

认证：环境变量 MX_APIKEY（Header: apikey）
配额：每日约 150 次调用；大盘复盘每次运行仅需 1~2 次（指数全量查询 + 缺失补查）

失败降级：返回 None 让 DataFetcherManager 继续走 Efinance/Akshare 兜底链。
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from .base import BaseFetcher, DataFetchError

logger = logging.getLogger(__name__)

_BASE_URL = "https://mkapi2.dfcfs.com/finskillshub/api/claw/query"
_REQUEST_TIMEOUT = 30

# (中文名, Tushare 代码, 市场前缀代码)
_CN_MAIN_INDICES = (
    ("上证指数", "000001.SH", "sh000001"),
    ("深证成指", "399001.SZ", "sz399001"),
    ("创业板指", "399006.SZ", "sz399006"),
    ("科创50", "000688.SH", "sh000688"),
    ("上证50", "000016.SH", "sh000016"),
    ("沪深300", "000300.SH", "sh000300"),
)

# 查询时附带的指标列，尽量在一次调用内拿全表格所需字段。
# 注意：妙想对 toolQuery 的措辞极敏感，多写「昨收」「涨跌额」会让它改吐沪深300
# 成分股表（几百行证券代码）而丢掉指数行情；「振幅」是实测安全的，勿再随手加词。
_INDICATOR_QUERY = "最新价 涨跌幅 今开 最高 最低 振幅 成交量 成交额"


def _match_cn_index(text: str) -> Optional[Tuple[str, str, str]]:
    """从标题/列名中匹配 6 个 A 股主要指数，返回 (name, ts_code, code)。"""
    for name, ts_code, code in _CN_MAIN_INDICES:
        if name in (text or ""):
            return name, ts_code, code
    return None


def _to_num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _to_pct(value: Any) -> float:
    """'-0.111%' -> -0.111"""
    if value is None:
        return 0.0
    s = str(value).strip().replace("%", "").replace("％", "").strip()
    if not s or s in ("-", "None", "nan"):
        return 0.0
    return _to_num(s)


def _to_point(value: Any) -> float:
    """'3952.179点' -> 3952.179；'1.256万' -> 12560.0（指数点位也会带中文单位）"""
    return _parse_cn_number(value, ("点",), {"万亿": 1e12, "亿": 1e8, "万": 1e4})


def _parse_cn_number(value: Any, unit_words: Tuple[str, ...], mult_map: Dict[str, float]) -> float:
    """解析 '5482亿' / '1.131万亿' / '630.3亿股' 这类带中文单位的数值。"""
    if value is None:
        return 0.0
    s = str(value).strip()
    for word in unit_words:
        s = s.replace(word, "")
    s = s.strip()
    if not s or s in ("-", "None", "nan"):
        return 0.0
    multiplier = 1.0
    for unit, mult in mult_map.items():
        if unit in s:
            multiplier = mult
            s = s.replace(unit, "")
            break
    return _to_num(s) * multiplier


def _parse_amount(value: Any) -> float:
    """成交额 -> 元"""
    return _parse_cn_number(value, ("元",), {"万亿": 1e12, "亿": 1e8, "万": 1e4})


def _parse_volume(value: Any) -> float:
    """成交量 -> 手（mx-data 返回 股，除以 100 转手）"""
    shares = _parse_cn_number(value, ("股",), {"万亿": 1e12, "亿": 1e8, "万": 1e4})
    return shares / 100.0


def _indicator_field(label: str) -> Optional[str]:
    """将指标中文名映射到标准字段名。"""
    if "涨跌幅" in label:
        return "change_pct"
    if "涨跌额" in label:
        return "change"
    if "最新价" in label or "收盘价" in label:
        return "current"
    if "今开" in label or "开盘" in label:
        return "open"
    if "最高" in label:
        return "high"
    if "最低" in label:
        return "low"
    if "成交额" in label:
        return "amount"
    if "成交量" in label:
        return "volume"
    if "振幅" in label:
        return "amplitude"
    return None


def _first_value(raw: Any) -> Any:
    if isinstance(raw, list):
        return raw[0] if raw else None
    return raw


class MxDataFetcher(BaseFetcher):
    """妙想金融数据：仅作为主要指数行情的优先数据源。"""

    name = "MxDataFetcher"
    priority = int(os.getenv("MX_DATA_PRIORITY", "-2"))  # 最高优先级，先于 Tushare(P-1)

    def __init__(self) -> None:
        self._api_key = (os.getenv("MX_APIKEY") or "").strip()

    def is_available_for_request(self, capability: str = "") -> bool:
        """仅服务于主要指数行情；日线/实时/名称等能力一律不参与。"""
        if capability in ("daily_data", "realtime_quote", "stock_name", "stock_list"):
            return False
        return bool(self._api_key)

    # ---- 不支持的能力：即时失败，避免被其他路径选用 ----
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str):
        raise DataFetchError("[MxDataFetcher] 不支持日线数据（仅提供主要指数行情）")

    def _normalize_data(self, df, stock_code: str):
        raise DataFetchError("[MxDataFetcher] 不支持日线数据（仅提供主要指数行情）")

    # ---- 主要指数行情 ----
    def get_main_indices(self, region: str = "cn") -> Optional[List[Dict[str, Any]]]:
        if region != "cn":
            return None
        if not self._api_key:
            logger.debug("[MxDataFetcher] MX_APIKEY 未配置，跳过指数查询")
            return None

        try:
            names = " ".join(name for name, _, _ in _CN_MAIN_INDICES)
            first = self._query_indices(f"{names} {_INDICATOR_QUERY}")

            missing = [name for name, _, _ in _CN_MAIN_INDICES if name not in first]
            if missing:
                logger.warning(
                    "[MxDataFetcher] 首次查询缺失 %s，执行补查",
                    ",".join(missing),
                )
                extra = self._query_indices(f"{' '.join(missing)} {_INDICATOR_QUERY}")
                for name, values in extra.items():
                    first.setdefault(name, values)

            results: List[Dict[str, Any]] = []
            for name, _ts_code, code in _CN_MAIN_INDICES:
                values = first.get(name)
                if not values:
                    logger.warning(
                        "[MxDataFetcher] 指数行情不完整，放弃本数据源让 fallback 继续"
                    )
                    return None
                results.append(self._build_index_dict(name, code, values))

            logger.info(
                "[MxDataFetcher] 获取指数行情成功 count=%d",
                len(results),
            )
            return results or None
        except Exception as exc:
            logger.warning("[MxDataFetcher] 获取指数行情失败: %s", exc)
            return None

    def _query_indices(self, tool_query: str) -> Dict[str, Dict[str, Any]]:
        """发起一次妙想查询，返回 {指数中文名: {字段: 原始值}}。"""
        headers = {"Content-Type": "application/json", "apikey": self._api_key}
        response = requests.post(
            _BASE_URL,
            headers=headers,
            json={"toolQuery": tool_query},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("status") != 0:
            logger.warning(
                "[MxDataFetcher] 接口状态异常 status=%s message=%s",
                result.get("status"),
                result.get("message", "")[:120],
            )
            return {}

        inner = (result.get("data") or {}).get("data") or {}
        dto_list = ((inner.get("searchDataResultDTO") or {}).get("dataTableDTOList")) or []
        parsed: Dict[str, Dict[str, Any]] = {}
        for dto in dto_list:
            self._parse_dto(dto, parsed)
        return parsed

    def _parse_dto(self, dto: Dict[str, Any], out: Dict[str, Dict[str, Any]]) -> None:
        table = dto.get("table") or {}
        if not isinstance(table, dict):
            return
        name_map = dto.get("nameMap") or {}
        if not isinstance(name_map, dict):
            name_map = {}
        keys = [key for key in table.keys() if key != "headName"]

        # 形态 A：列名为指数名（一次查询多指数、单一指标）
        shape_a_match = next((key for key in keys if _match_cn_index(str(key))), None)
        if shape_a_match is not None:
            field = self._shape_a_field(name_map)
            if field is None:
                return
            for key in keys:
                matched = _match_cn_index(str(key))
                if not matched:
                    continue
                name = matched[0]
                out.setdefault(name, {})[field] = _first_value(table.get(key))
            return

        # 形态 B：行名（headName）为指数名、列为指标 —— 一次查询拿多指数多指标。
        # 关键：指数名在 headName 的「值」里，不在列名里，所以上面按列名的探测
        # 认不出来。漏了这一支时每个 DTO 只会按 title 取第 0 行，同表其余指数被
        # 静默丢弃（曾导致开盘/最高/最低等只剩 1/6 的指数有值）。
        head_names = table.get("headName")
        if isinstance(head_names, list) and any(
            _match_cn_index(str(head)) for head in head_names
        ):
            for key in keys:
                field = _indicator_field(str(name_map.get(str(key), str(key))))
                if field is None:
                    continue
                column = table.get(key)
                if not isinstance(column, list):
                    continue
                for row, head in enumerate(head_names):
                    matched_row = _match_cn_index(str(head))
                    if not matched_row or row >= len(column):
                        continue
                    out.setdefault(matched_row[0], {})[field] = column[row]
            return

        # 形态 C：单指数（标题带指数名），列为指标编码
        matched = _match_cn_index(dto.get("title") or dto.get("entityName") or "")
        if not matched:
            return
        name = matched[0]
        values = out.setdefault(name, {})
        for key in keys:
            label = str(name_map.get(str(key), str(key)))
            field = _indicator_field(label)
            if field is None:
                continue
            values[field] = _first_value(table.get(key))

    @staticmethod
    def _shape_a_field(name_map: Dict[str, Any]) -> Optional[str]:
        """形态 A 下从 nameMap 推断查询的单一指标字段。"""
        for key, label in name_map.items():
            if str(key) == "headNameSub":
                continue
            field = _indicator_field(str(label))
            if field is not None:
                return field
        return None

    @staticmethod
    def _build_index_dict(name: str, code: str, values: Dict[str, Any]) -> Dict[str, Any]:
        """将原始值按标准字段组装成 MarketIndex 兼容的字典。"""
        current = _to_point(values.get("current"))
        return {
            "code": code,
            "name": name,
            "current": current,
            "change": _to_num(values.get("change")),
            "change_pct": _to_pct(values.get("change_pct")),
            "open": _to_point(values.get("open")),
            "high": _to_point(values.get("high")),
            "low": _to_point(values.get("low")),
            "prev_close": 0.0,
            "volume": _parse_volume(values.get("volume")),
            "amount": _parse_amount(values.get("amount")),
            "amplitude": _to_pct(values.get("amplitude")),
        }
