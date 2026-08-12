# -*- coding: utf-8 -*-
"""Regression tests for chip distribution provider fallback."""

import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from data_provider.base import (
    DataFetcherManager,
    _is_chip_connection_error,
    _load_chip_cache,
    _save_chip_cache,
)
from data_provider.realtime_types import ChipDistribution, get_chip_circuit_breaker


@pytest.fixture(autouse=True)
def _isolate_chip_cache(tmp_path):
    """把筹码磁盘缓存重定向到临时目录，避免测试污染真实 data/cache/chip/。"""
    with patch("data_provider.base._CHIP_CACHE_DIR", tmp_path):
        yield


class _ChipFetcher:
    def __init__(self, name: str, priority: int, result):
        self.name = name
        self.priority = priority
        self._result = result
        self.calls = 0

    def get_chip_distribution(self, stock_code: str):
        self.calls += 1
        return self._result


class _FailingChipFetcher(_ChipFetcher):
    def __init__(self, name: str, priority: int, error: Exception):
        super().__init__(name, priority, None)
        self._error = error

    def get_chip_distribution(self, stock_code: str):
        self.calls += 1
        raise self._error


def _run_with_chip_diagnostics(manager: DataFetcherManager):
    from src.services.run_diagnostics import (
        activate_run_diagnostic_context,
        current_diagnostic_snapshot,
        reset_run_diagnostic_context,
    )

    flow_events = []
    token = activate_run_diagnostic_context(
        trace_id="trace-chip",
        task_id="task-chip",
        query_id="query-chip",
        stock_code="600519",
        trigger_source="api",
        event_sink=flow_events.append,
    )
    try:
        with patch("src.config.get_config", return_value=SimpleNamespace(enable_chip_distribution=True)):
            chip = manager.get_chip_distribution("600519")
        diagnostics = current_diagnostic_snapshot()
    finally:
        reset_run_diagnostic_context(token)
    return chip, diagnostics, flow_events


def test_manager_skips_placeholder_chip_distribution_and_tries_next_fetcher():
    get_chip_circuit_breaker().reset()
    empty_chip = ChipDistribution(code="600519")
    valid_chip = ChipDistribution(
        code="600519",
        profit_ratio=0.61,
        avg_cost=12.3,
        concentration_90=0.13,
    )
    manager = DataFetcherManager(
        fetchers=[
            _ChipFetcher("EmptyFetcher", 0, empty_chip),
            _ChipFetcher("ValidFetcher", 1, valid_chip),
        ]
    )

    chip, diagnostics, flow_events = _run_with_chip_diagnostics(manager)

    assert chip is valid_chip
    assert diagnostics is not None
    provider_runs = diagnostics["provider_runs"]
    assert [run["data_type"] for run in provider_runs] == ["chip", "chip"]
    assert [run["success"] for run in provider_runs] == [False, True]
    assert provider_runs[0]["fallback_to"] == "ValidFetcher"
    assert provider_runs[0]["record_count"] == 0
    assert provider_runs[1]["record_count"] == 1
    assert [event["type"] for event in flow_events] == [
        "provider_run_started",
        "provider_run",
        "provider_run_started",
        "provider_run",
    ]
    assert flow_events[0]["node_id"] == flow_events[1]["node_id"]
    assert flow_events[2]["node_id"] == flow_events[3]["node_id"]
    assert flow_events[0]["node_id"] == "provider_chip_emptyfetcher_1"
    assert flow_events[2]["node_id"] == "provider_chip_validfetcher_2"


def test_manager_accepts_zero_concentration_chip_distribution():
    get_chip_circuit_breaker().reset()
    zero_concentration_chip = ChipDistribution(
        code="600519",
        profit_ratio=0.61,
        avg_cost=12.3,
        concentration_90=0.0,
        concentration_70=0.0,
    )
    fallback_chip = ChipDistribution(
        code="600519",
        profit_ratio=0.62,
        avg_cost=12.5,
        concentration_90=0.13,
    )
    zero_fetcher = _ChipFetcher("ZeroConcentrationFetcher", 0, zero_concentration_chip)
    fallback_fetcher = _ChipFetcher("FallbackFetcher", 1, fallback_chip)
    manager = DataFetcherManager(fetchers=[zero_fetcher, fallback_fetcher])

    chip, diagnostics, flow_events = _run_with_chip_diagnostics(manager)

    assert chip is zero_concentration_chip
    assert zero_fetcher.calls == 1
    assert fallback_fetcher.calls == 0
    assert diagnostics is not None
    assert len(diagnostics["provider_runs"]) == 1
    assert diagnostics["provider_runs"][0]["data_type"] == "chip"
    assert diagnostics["provider_runs"][0]["success"] is True
    assert [event["type"] for event in flow_events] == ["provider_run_started", "provider_run"]
    assert flow_events[0]["node_id"] == flow_events[1]["node_id"]


def test_manager_records_failed_chip_attempt_and_falls_back_to_next_fetcher():
    get_chip_circuit_breaker().reset()
    valid_chip = ChipDistribution(
        code="600519",
        profit_ratio=0.61,
        avg_cost=12.3,
        concentration_90=0.13,
    )
    failing_fetcher = _FailingChipFetcher("FailingFetcher", 0, RuntimeError("temporary chip failure"))
    fallback_fetcher = _ChipFetcher("FallbackFetcher", 1, valid_chip)
    manager = DataFetcherManager(fetchers=[failing_fetcher, fallback_fetcher])

    chip, diagnostics, flow_events = _run_with_chip_diagnostics(manager)

    assert chip is valid_chip
    assert failing_fetcher.calls == 1
    assert fallback_fetcher.calls == 1
    assert diagnostics is not None
    provider_runs = diagnostics["provider_runs"]
    assert [run["provider"] for run in provider_runs] == ["FailingFetcher", "FallbackFetcher"]
    assert provider_runs[0]["success"] is False
    assert provider_runs[0]["error_type"] == "RuntimeError"
    assert provider_runs[0]["fallback_to"] == "FallbackFetcher"
    assert provider_runs[1]["success"] is True
    assert [event["type"] for event in flow_events] == [
        "provider_run_started",
        "provider_run",
        "provider_run_started",
        "provider_run",
    ]


def test_success_writes_last_known_good_cache():
    get_chip_circuit_breaker().reset()
    valid_chip = ChipDistribution(
        code="600519",
        date="2026-08-11",
        profit_ratio=0.61,
        avg_cost=12.3,
        concentration_90=0.13,
    )
    manager = DataFetcherManager(fetchers=[_ChipFetcher("ValidFetcher", 0, valid_chip)])

    with patch("src.config.get_config", return_value=SimpleNamespace(enable_chip_distribution=True)):
        chip = manager.get_chip_distribution("600519")

    assert chip is valid_chip
    cached = _load_chip_cache("600519")
    assert cached is not None
    assert cached.date == "2026-08-11"
    assert cached.profit_ratio == 0.61
    assert cached.avg_cost == 12.3
    assert cached.concentration_90 == 0.13


def test_failure_falls_back_to_cached_chip_without_retry():
    get_chip_circuit_breaker().reset()
    cached_chip = ChipDistribution(
        code="600519",
        date="2026-08-10",
        profit_ratio=0.55,
        avg_cost=11.8,
        concentration_90=0.14,
    )
    _save_chip_cache("600519", cached_chip)
    failing_fetcher = _FailingChipFetcher("FailingFetcher", 0, ConnectionError("Remote end closed"))
    manager = DataFetcherManager(fetchers=[failing_fetcher])

    with patch("src.config.get_config", return_value=SimpleNamespace(enable_chip_distribution=True)):
        chip = manager.get_chip_distribution("600519")

    assert chip is not None
    assert chip.date == "2026-08-10"
    # 有可用缓存时不烧补漏重试的时间，只请求一次
    assert failing_fetcher.calls == 1


def test_connection_error_triggers_retry_when_no_cache():
    get_chip_circuit_breaker().reset()
    valid_chip = ChipDistribution(
        code="600519",
        profit_ratio=0.61,
        avg_cost=12.3,
        concentration_90=0.13,
    )

    class _FlakyFetcher(_ChipFetcher):
        def get_chip_distribution(self, stock_code):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("Remote end closed connection without response")
            return self._result

    flaky = _FlakyFetcher("FlakyFetcher", 0, valid_chip)
    manager = DataFetcherManager(fetchers=[flaky])

    with patch("data_provider.base._CHIP_RETRY_COOLDOWN_SECONDS", 0.0), patch(
        "src.config.get_config", return_value=SimpleNamespace(enable_chip_distribution=True)
    ):
        chip = manager.get_chip_distribution("600519")

    assert chip is valid_chip
    assert flaky.calls == 2


def test_expired_cache_returns_none(tmp_path):
    _save_chip_cache(
        "600519",
        ChipDistribution(code="600519", profit_ratio=0.5, avg_cost=10.0, concentration_90=0.12),
    )
    path = tmp_path / "600519.json"
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    d["saved_at"] = time.time() - 8 * 24 * 3600  # 8 天前，超过 7 天阈值
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)

    assert _load_chip_cache("600519") is None


def test_is_chip_connection_error_detection():
    assert _is_chip_connection_error("ConnectionError", "Remote end closed connection without response")
    assert _is_chip_connection_error("RetryError", "RetryError[... state=finished raised ConnectionError]")
    assert _is_chip_connection_error("Timeout", "timed out")
    assert not _is_chip_connection_error("RuntimeError", "no permission")
    assert not _is_chip_connection_error("", "empty or incomplete chip distribution")
