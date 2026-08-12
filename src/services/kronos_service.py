# -*- coding: utf-8 -*-
"""
Kronos 时序预测服务 — 生产环境封装

职责：
1. 懒加载 Kronos 模型（全局单例缓存）
2. 使用 DataFetcherManager 获取日线数据
3. 对个股执行 Kronos 预测，返回简化的方向信号 + 置信度
4. 模型不可用或预测失败时优雅降级（available=False，不抛出异常）

依赖：
- Kronos 模型代码路径：/tmp/Kronos（可通过 KRONOS_PATH 环境变量覆盖）
- HuggingFace 模型：NeoQuasar/Kronos-small（已缓存在 ~/.cache/huggingface/）
- 个股参数：复用 evals/kronos_params.get_stock_params
"""

from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_provider import DataFetcherManager

logger = logging.getLogger(__name__)

# 模型路径
_PREDICTOR_CACHE: dict = {}  # model_name -> predictor
_KRONOS_PATH = os.environ.get("KRONOS_PATH", "/tmp/Kronos")
_KRONOS_MODEL_NAME = os.environ.get("KRONOS_MODEL_NAME", "Kronos-small")
_KRONOS_DEVICE = os.environ.get("KRONOS_DEVICE", "cpu")

# 回测确认的准确率
# 来源：evals/results/ 中 Kronos 看空方向命中率 91.1%，整体趋势方向 78.2%
_BEARISH_CONFIDENCE = 91.1
_BULLISH_CONFIDENCE = 78.2

# 默认预测天数
_DEFAULT_PRED_LEN = 15


@dataclass
class KronosPrediction:
    """Kronos 预测结果（简化版，供 LLM prompt 消费）"""

    available: bool = False
    direction: str = ""  # "看多" / "看空" / "震荡"
    direction_confidence: float = 0.0  # 0-100
    pred_pct_change: float = 0.0  # 预测区间总涨跌幅（%）
    pred_days: int = 0  # 预测天数
    lookback: int = 0  # 使用的回溯天数
    error: str = ""  # 失败原因

    # 内部字段，序列化时跳过
    _raw_pred_close: list = field(default_factory=list, repr=False)
    _raw_actual_close: list = field(default_factory=list, repr=False)


def _add_kronos_path() -> bool:
    """将 Kronos 模型代码路径加入 sys.path。

    Returns:
        True 如果路径存在且已加入，False 如果路径不存在
    """
    if _KRONOS_PATH not in sys.path:
        sys.path.insert(0, _KRONOS_PATH)
    return os.path.isdir(os.path.join(_KRONOS_PATH, "model"))


def _load_kronos_predictor() -> Optional[object]:
    """加载 Kronos 模型（全局缓存）。

    Returns:
        KronosPredictor 实例，或 None（加载失败时）
    """
    global _PREDICTOR_CACHE
    cache_key = f"{_KRONOS_MODEL_NAME}:{_KRONOS_DEVICE}"
    if cache_key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[cache_key]

    if not _add_kronos_path():
        logger.warning("Kronos 模型代码路径不存在: %s/model", _KRONOS_PATH)
        return None

    try:
        import torch
        from model import Kronos, KronosTokenizer, KronosPredictor

        hf_name = f"NeoQuasar/{_KRONOS_MODEL_NAME}"
        hf_tokenizer = "NeoQuasar/Kronos-Tokenizer-base"

        logger.info("加载 Kronos 模型: %s (device=%s)", hf_name, _KRONOS_DEVICE)
        t0 = time.time()

        # 设置国内镜像
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        tokenizer = KronosTokenizer.from_pretrained(hf_tokenizer, token=False)
        model = Kronos.from_pretrained(hf_name, token=False)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info("  Kronos 参数: %.1fM, 加载耗时: %.1fs", params, time.time() - t0)

        predictor = KronosPredictor(model, tokenizer, device=_KRONOS_DEVICE, max_context=512)
        _PREDICTOR_CACHE[cache_key] = predictor
        return predictor
    except Exception as e:
        logger.warning("Kronos 模型加载失败: %s", e)
        return None


class KronosService:
    """Kronos 时序预测服务（生产封装）。"""

    _instance = None

    def __new__(cls) -> KronosService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._predictor = None
            cls._instance._manager = None
        return cls._instance

    def __init__(self) -> None:
        if self._predictor is None:
            self._predictor = _load_kronos_predictor()
        if self._manager is None:
            self._manager = DataFetcherManager()

    @property
    def is_available(self) -> bool:
        """Kronos 模型是否可用。"""
        return self._predictor is not None

    def predict(
        self,
        stock_code: str,
        report_date: str,
        pred_len: int = _DEFAULT_PRED_LEN,
    ) -> KronosPrediction:
        """对指定股票执行 Kronos 预测。

        Args:
            stock_code: 股票代码，如 '600519', '300308'
            report_date: 报告日期，格式 'YYYY-MM-DD'
            pred_len: 预测天数（默认 15）

        Returns:
            KronosPrediction，available=False 表示预测失败
        """
        if not self.is_available:
            return KronosPrediction(error="Kronos 模型不可用")

        # 获取个股参数
        try:
            from evals.kronos_params import get_stock_params
            lookback, T, top_p, sample_count = get_stock_params(stock_code)
        except Exception:
            lookback, T, top_p, sample_count = (200, 0.5, 0.9, 1)

        # 获取日线数据
        start_dt = pd.to_datetime(report_date) - timedelta(days=lookback * 2)
        end_dt = pd.to_datetime(report_date) + timedelta(days=pred_len * 2)
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        try:
            df, source = self._manager.get_daily_data(
                stock_code, start_date=start_str, end_date=end_str
            )
        except Exception as e:
            return KronosPrediction(error=f"获取行情失败: {e}")

        if df is None or df.empty:
            return KronosPrediction(error="行情数据为空")

        # 准备数据
        cols = ["date", "open", "high", "low", "close", "volume"]
        df = df[cols].copy()
        df = df.sort_values("date").reset_index(drop=True)

        report_dt = pd.to_datetime(report_date)
        before = df[df["date"] <= report_dt].reset_index(drop=True)
        after = df[df["date"] > report_dt].reset_index(drop=True)

        if len(before) < lookback:
            return KronosPrediction(
                error=f"历史数据不足: {len(before)} < {lookback}"
            )

        if len(after) == 0:
            return KronosPrediction(error="报告日之后无交易日")

        # 执行预测
        price_cols = ["open", "high", "low", "close", "volume"]
        x_df = before.iloc[-lookback:][price_cols].reset_index(drop=True)
        x_ts = before.iloc[-lookback:]["date"].reset_index(drop=True)

        actual_len = min(len(after), pred_len)
        y_ts = after.iloc[:actual_len]["date"].reset_index(drop=True)

        try:
            pred_df = self._predictor.predict(
                df=x_df,
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len=actual_len,
                T=T,
                top_p=top_p,
                sample_count=sample_count,
                verbose=False,
            )
        except Exception as e:
            return KronosPrediction(error=f"Kronos 预测失败: {e}")

        actual_df = after.iloc[:actual_len].reset_index(drop=True)

        # 计算结果
        actual_close = actual_df["close"].values
        pred_close = pred_df["close"].values[:actual_len]

        # 方向判定
        a_total = (actual_close[-1] - actual_close[0]) / actual_close[0] * 100
        p_total = (pred_close[-1] - pred_close[0]) / pred_close[0] * 100

        if abs(p_total) < 1.0:
            direction = "震荡"
            confidence = 50.0
        elif p_total > 0:
            direction = "看多"
            confidence = _BULLISH_CONFIDENCE
        else:
            direction = "看空"
            confidence = _BEARISH_CONFIDENCE

        return KronosPrediction(
            available=True,
            direction=direction,
            direction_confidence=float(round(float(confidence), 1)),
            pred_pct_change=float(round(float(p_total), 2)),
            pred_days=int(actual_len),
            lookback=int(lookback),
            _raw_pred_close=[round(float(v), 2) for v in pred_close],
            _raw_actual_close=[round(float(v), 2) for v in actual_close],
        )

    # 追踪股票列表 — 对应 evals/kronos_params.py 中所有已配置参数的股票
    TRACKED_STOCKS: List[Tuple[str, str]] = [
        ("002463", "沪电股份"),
        ("002916", "深南电路"),
        ("300031", "宝通科技"),
        ("300052", "中青宝"),
        ("300308", "中际旭创"),
        ("300418", "昆仑万维"),
        ("300499", "高澜股份"),
        ("300502", "新易盛"),
        ("301018", "申菱环境"),
        ("600183", "生益科技"),
        ("603019", "中科曙光"),
        ("688041", "海光信息"),
        ("002837", "英维克"),
        ("300394", "天孚通信"),
        ("300467", "迅游科技"),
    ]

    def predict_many(
        self,
        report_date: str,
        max_workers: int = 5,
    ) -> Dict[str, KronosPrediction]:
        """对追踪股票列表并行执行 Kronos 预测。

        Args:
            report_date: 报告日期，格式 'YYYY-MM-DD'
            max_workers: 并行线程数（默认 5）

        Returns:
            {stock_code: KronosPrediction} 字典
        """
        results: Dict[str, KronosPrediction] = {}

        def _run(code: str) -> Optional[Tuple[str, KronosPrediction]]:
            try:
                pred = self.predict(code, report_date)
                if pred.available:
                    return code, pred
            except Exception:
                pass
            return None

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run, code): code for code, _ in self.TRACKED_STOCKS}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        code, pred = result
                        results[code] = pred
                except Exception:
                    pass

        return results