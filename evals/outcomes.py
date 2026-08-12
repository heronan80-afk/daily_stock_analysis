"""给定信号 + 窗口，取后续行情并判定 hit/miss。

命中判定（horizon 个交易日窗口）：
    long  (看多/买入): T+1~T+horizon 内任一收盘价 > T 收盘价  -> hit
    short (看空/卖出): T+1~T+horizon 内任一收盘价 < T 收盘价  -> hit

T = 报告日期的下一个交易日（报告盘后生成，以 T 日收盘为基准）。
基准价 = T 日收盘价。窗口内取不到行情 -> skipped。
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .extract_signals import Signal

logger = logging.getLogger(__name__)


class CachedFetcher:
    """按股票代码缓存行情，规避同一股票在多份报告里重复拉取。

    同一只股票在多份报告里会重复出现（15 只股票 × ~18 份报告 ≈ 250 次请求），
    而每只股票实际只需要一次宽窗口的行情。本缓存按 code 维度存储已取到的
    完整 DataFrame 及其**请求范围**，后续请求按日期在本地切片返回；当请求范围
    超出已缓存的请求范围时，才对底层 fetcher 发起一次更宽的请求并替换缓存。

    注意：覆盖判定基于"请求范围"而非"返回数据日期"——行情数据按交易日对齐，
    返回的最小/最大日期常与请求边界不同（周末/节假日），用返回日期判覆盖会误判未命中。

    last_call_cached 标记上一次 get_daily_data 是否命中缓存（未发起网络请求），
    供调用方决定是否限流 sleep。
    """

    def __init__(self, fetcher):
        self._fetcher = fetcher
        # code -> (req_start, req_end, df)；req_start/req_end 是已缓存覆盖的请求范围
        self._cache: Dict[str, Tuple[Optional[str], Optional[str], Optional[pd.DataFrame]]] = {}
        self.hits = 0
        self.misses = 0
        self.last_call_cached: bool = False

    def _slice(self, df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """按日期在本地切片（date 列为字符串 YYYY-MM-DD）。"""
        out = df
        if start_date is not None:
            out = out[out["date"] >= start_date]
        if end_date is not None:
            out = out[out["date"] <= end_date]
        return out.copy()

    def get_daily_data(self, code: str, start_date: str = None, end_date: str = None, **kwargs):
        entry = self._cache.get(code)
        if entry is not None:
            cached_start, cached_end, df = entry
            if df is not None and self._covers(cached_start, cached_end, start_date, end_date):
                self.hits += 1
                self.last_call_cached = True
                return self._slice(df, start_date, end_date)

        # 缓存未命中或覆盖不全 -> 发起一次更宽的请求，向现有缓存范围外扩展
        self.misses += 1
        self.last_call_cached = False
        if entry is not None and entry[2] is not None:
            cached_start, cached_end, _ = entry
            new_start = min(filter(None, [cached_start, start_date])) if (cached_start or start_date) else None
            new_end = max(filter(None, [cached_end, end_date])) if (cached_end or end_date) else None
        else:
            new_start, new_end = start_date, end_date
        df = self._fetcher.get_daily_data(code, start_date=new_start, end_date=new_end, **kwargs)
        if df is not None and not df.empty and "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)
            df["date"] = df["date"].astype(str).str[:10]
        # 缓存覆盖范围以"请求范围"为准（非返回数据日期），避免交易日对齐导致误判
        self._cache[code] = (new_start, new_end, df if df is not None else None)
        return self._slice(df, start_date, end_date) if (df is not None and not df.empty and "date" in df.columns) else df

    @staticmethod
    def _covers(cached_start, cached_end, start, end) -> bool:
        if cached_start is None or cached_end is None:
            return start is None and end is None
        if start is not None and start < cached_start:
            return False
        if end is not None and end > cached_end:
            return False
        return True

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


@dataclass
class Outcome:
    """单个信号回测结果。

    两套命中口径并存：
    - result (宽松)：窗口内任一收盘满足方向即 hit（已实现峰值/谷值）
    - result_strict (严格)：窗口末日收盘满足方向才 hit（期末方向）
    """

    signal: Signal
    baseline_date: Optional[str]  # T 日（基准日，报告日之后的第一个有行情的交易日）
    baseline_close: Optional[float]  # T 日收盘价
    window_closes: List[float]  # T+1 ~ T+horizon 收盘价序列
    window_dates: List[str]  # 对应日期
    max_pct_change: Optional[float]  # 窗口内相对基准的最大涨/跌幅（long 取最大涨幅，short 取最大跌幅，宽松口径用）
    end_pct_change: Optional[float]  # 窗口末日收盘相对基准的涨/跌幅（严格口径用）
    result: str  # 宽松: hit / miss / pending / skipped / no_direction
    result_strict: str  # 严格: hit / miss / pending / skipped / no_direction
    reason: str  # skipped/no_direction 时的原因

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signal"] = self.signal.to_dict()
        return d


def _next_trading_days(dates: List[str], baseline_idx: int, horizon: int) -> List[int]:
    """从 baseline_idx 之后取 horizon 个交易日的索引。"""
    return list(range(baseline_idx + 1, min(baseline_idx + 1 + horizon, len(dates))))


def _signal_request_range(signal: Signal, horizon: int, extra_lookback: int) -> Tuple[str, str]:
    """计算单个信号需要的行情请求范围，与 compute_outcome 内一致。"""
    report_dt = datetime.strptime(signal.report_date, "%Y-%m-%d").date()
    end_date = (report_dt + timedelta(days=horizon * 2 + 15)).strftime("%Y-%m-%d")
    start_date = (report_dt - timedelta(days=extra_lookback)).strftime("%Y-%m-%d")
    return start_date, end_date


def compute_outcome(
    signal: Signal,
    fetcher,
    horizon: int = 5,
    extra_lookback: int = 10,
) -> Outcome:
    """计算单个信号的回测结果。

    fetcher: 已实例化的 AkshareFetcher（或任何带 get_daily_data 的 fetcher）。
    extra_lookback: 报告日之前多取的日历日，确保能覆盖到报告日附近的交易日。
    """
    # 无方向性信号（震荡/持有/观望）不取行情、不判定，直接记 no_direction
    direction = signal.effective_direction
    if direction not in ("long", "short"):
        return _no_direction(signal, f"无方向性(action={signal.action_kind}, direction={signal.direction})")

    report_dt = datetime.strptime(signal.report_date, "%Y-%m-%d").date()
    # 窗口上限：报告日 + horizon 交易日 + 日历缓冲（周末/节假日）
    start_date, end_date = _signal_request_range(signal, horizon, extra_lookback)

    try:
        df = fetcher.get_daily_data(signal.code, start_date=start_date, end_date=end_date)
    except Exception as e:
        return _skipped(signal, f"取行情失败: {e}")

    if df is None or df.empty:
        return _skipped(signal, "返回空行情")

    # 标准化列（get_daily_data 已标准化，列名为 STANDARD_COLUMNS）
    if "date" not in df.columns or "close" not in df.columns:
        return _skipped(signal, f"行情缺 date/close 列: {list(df.columns)}")

    df = df.sort_values("date").reset_index(drop=True)
    # 日期统一为字符串 YYYY-MM-DD
    df["date"] = df["date"].astype(str).str[:10]

    # 找基准日：报告日之后的第一个交易日
    report_str = signal.report_date
    future = df[df["date"] > report_str]
    if future.empty:
        return _skipped(signal, "报告日之后无交易日行情（可能报告日过近）")
    baseline_idx = future.index[0]
    baseline_date = df.loc[baseline_idx, "date"]
    baseline_close = float(df.loc[baseline_idx, "close"])

    # 窗口：基准日之后的 horizon 个交易日
    window_idx = _next_trading_days(df["date"].tolist(), int(baseline_idx), horizon)
    if not window_idx:
        return _skipped(signal, "基准日之后无窗口交易日")

    window_rows = df.loc[window_idx]
    window_closes = [float(x) for x in window_rows["close"].tolist()]
    window_dates = [str(x) for x in window_rows["date"].tolist()]

    direction = signal.effective_direction
    pct_changes = [(c - baseline_close) / baseline_close for c in window_closes]

    if direction == "long":
        # 看多：窗口内任一收盘 > 基准 即 hit
        max_pct = max(pct_changes)
        result = "hit" if max_pct > 0 else "miss"
    else:  # short
        # 看空：窗口内任一收盘 < 基准 即 hit
        max_pct = min(pct_changes)
        result = "hit" if max_pct < 0 else "miss"

    # 窗口未满（报告太近、后续交易日还没走完）：hit 仍算命中，miss 转 pending 不计入分母
    if len(window_closes) < horizon and result == "miss":
        return Outcome(
            signal=signal,
            baseline_date=baseline_date,
            baseline_close=baseline_close,
            window_closes=window_closes,
            window_dates=window_dates,
            max_pct_change=max_pct,
            end_pct_change=pct_changes[-1],
            result="pending",
            result_strict="pending",  # 严格口径需末日收盘，窗口未满一律 pending
            reason=f"窗口未满({len(window_closes)}/{horizon})，暂无命中，待后续交易日",
        )

    # 严格口径：窗口末日收盘 vs 基准
    end_pct = pct_changes[-1]
    if len(window_closes) < horizon:
        result_strict = "pending"  # 窗口未满，末日未定
    elif direction == "long":
        result_strict = "hit" if end_pct > 0 else "miss"
    else:  # short
        result_strict = "hit" if end_pct < 0 else "miss"

    return Outcome(
        signal=signal,
        baseline_date=baseline_date,
        baseline_close=baseline_close,
        window_closes=window_closes,
        window_dates=window_dates,
        max_pct_change=max_pct,
        end_pct_change=end_pct,
        result=result,
        result_strict=result_strict,
        reason="",
    )


def _skipped(signal: Signal, reason: str) -> Outcome:
    return Outcome(
        signal=signal,
        baseline_date=None,
        baseline_close=None,
        window_closes=[],
        window_dates=[],
        max_pct_change=None,
        end_pct_change=None,
        result="skipped",
        result_strict="skipped",
        reason=reason,
    )


def _no_direction(signal: Signal, reason: str) -> Outcome:
    return Outcome(
        signal=signal,
        baseline_date=None,
        baseline_close=None,
        window_closes=[],
        window_dates=[],
        max_pct_change=None,
        end_pct_change=None,
        result="no_direction",
        result_strict="no_direction",
        reason=reason,
    )


def compute_outcomes(
    signals: List[Signal],
    fetcher,
    horizon: int = 5,
    sleep_per_call: float = 0.5,
    extra_lookback: int = 10,
) -> List[Outcome]:
    """批量计算，每只股票取行情后 sleep 限流。

    自动用 CachedFetcher 包一层，并预先按 code 拉取每只股票覆盖所有相关报告的
    宽窗口行情（min_start ~ max_end）。这样同一只股票在多份报告里只拉一次行情，
    后续 compute_outcome 的请求全部命中缓存、不 sleep、不请求。
    """
    cached = fetcher if isinstance(fetcher, CachedFetcher) else CachedFetcher(fetcher)

    # 预热：按 code 聚合所有方向性信号的请求范围，取并集一次性拉取
    code_ranges: Dict[str, Tuple[str, str]] = {}
    for sig in signals:
        if sig.effective_direction not in ("long", "short"):
            continue
        s, e = _signal_request_range(sig, horizon, extra_lookback)
        if sig.code in code_ranges:
            ps, pe = code_ranges[sig.code]
            code_ranges[sig.code] = (min(ps, s), max(pe, e))
        else:
            code_ranges[sig.code] = (s, e)
    n_codes = len(code_ranges)
    logger.info("预热行情缓存: %d 只股票，并行拉取宽窗口 (workers=5)", n_codes)
    with ThreadPoolExecutor(max_workers=5) as executor:
        fut_map = {
            executor.submit(cached.get_daily_data, code, start_date=s, end_date=e): code
            for code, (s, e) in sorted(code_ranges.items())
        }
        for i, future in enumerate(as_completed(fut_map), 1):
            code = fut_map[future]
            try:
                future.result()
            except Exception as e:
                logger.error("预热拉取失败 %s: %s", code, e)
            if i % 5 == 0:
                logger.info("预热进度 %d/%d", i, n_codes)
    # 并行拉取后统一 sleep 一轮，避免后续请求过快触发限流
    if sleep_per_call > 0:
        time.sleep(sleep_per_call)

    out: List[Outcome] = []
    for i, sig in enumerate(signals):
        oc = compute_outcome(sig, cached, horizon=horizon, extra_lookback=extra_lookback)
        out.append(oc)
        if oc.result == "skipped":
            logger.warning("skipped %s(%s) on %s: %s", sig.name, sig.code, sig.report_date, oc.reason)
        # 预热后正常信号全部命中缓存；这里保留 sleep 分支以防缓存未覆盖的边角情况
        if sleep_per_call > 0 and oc.result not in ("no_direction",) and not cached.last_call_cached:
            time.sleep(sleep_per_call)
        if (i + 1) % 10 == 0:
            logger.info("进度 %d/%d (缓存 hit=%d miss=%d)", i + 1, len(signals), cached.hits, cached.misses)
    logger.info(
        "行情缓存统计: hit=%d miss=%d 命中率=%.1f%% (预热拉取 %d 只股票)",
        cached.hits, cached.misses, cached.hit_rate * 100, n_codes,
    )
    return out
