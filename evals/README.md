# Evals - 决策报告回测

把 `reports/report_*.md` 决策仪表盘里的个股方向性信号，对照 N 个交易日后的实际行情，算命中率。
这是项目当前缺失的"agent 输出准不准"评测闭环（现有 `tests/` 是单元/集成测试，不评测预测质量）。

## 快速开始

```bash
cd daily_stock_analysis
./venv/bin/python -m evals.run_eval --horizon 5
```

需要网络（akshare 取行情）。结果写到 `evals/results/`：
- `outcomes_<ts>.jsonl` - 逐条信号回测结果（信号 + 基准价 + 窗口收盘价 + hit/miss）
- `summary_<ts>.md` - 分桶命中率统计表

调试（只跑前 3 个信号，带日志）：
```bash
./venv/bin/python -m evals.run_eval --limit 3 -v
```

## 命中判定规则

T = 报告日期的**下一个交易日**（报告盘后生成，以 T 日收盘为基准）。

| 信号 | 判 hit | 判 miss |
|------|--------|---------|
| 看多/买入 (long) | T+1~T+horizon 内任一收盘价 > T 收盘 | 窗口内从未高于 T 收盘 |
| 看空/卖出/减仓 (short) | T+1~T+horizon 内任一收盘价 < T 收盘 | 窗口内从未低于 T 收盘 |
| 震荡/观望/持有 | 不纳入命中率（记为 `no_direction`） | - |

两套口径并存（结果同时落 `result` 与 `result_strict`）：
- **宽松** `result`：窗口内任一收盘满足方向即 hit（已实现峰值/谷值）。
- **严格** `result_strict`：窗口末日收盘满足方向才 hit（期末方向）。窗口未满一律 `pending`。
- 对照两口径可分离"规则红利"（冲高/探底后回落）与真实预测能力。strict hit 必然蕴含 loose hit。

- 取不到行情 / 报告日之后无交易日 → 记为 `skipped`，不计入分母
- 报告太近、窗口未走满（`len(窗口) < horizon`）：已命中算 `hit`（命中单调，不会反转）；暂未命中算 `pending`，不计入分母，待后续交易日再判
- 命中率 = hit / (hit + miss)

## 结果分桶

`summary` 按 5 个维度分桶：方向（long/short）、操作类型、评分区间（≥60 / 45-59 / <45）、报告日期、skipped 原因。

## 设计要点

- **复用项目数据源**：行情走 `data_provider/akshare_fetcher.py` 的 `get_daily_data`（akshare 三级 fallback em/sina/tencent），不直接调 akshare，保证和主流程一致的清洗/标准化。
- **按 code 缓存行情**：`CachedFetcher` 按 code 维度缓存宽窗口行情，`compute_outcomes` 预热阶段把每只股票跨所有报告的请求范围取并集一次性拉取。15 只股票 × ~18 份报告的回测从 ~30min 降到 ~3min（akshare 请求减少 ~90%）。覆盖判定基于请求范围而非返回日期（akshare 按交易日对齐，返回日期与请求边界常不一致，用返回日期判覆盖会误判未命中）。
- **不调 LLM**：evals 是确定性回测，不烧 token（符合 Inference Economics）。
- **不碰现有代码**：全部在 `evals/` 下，不改 `main.py` / `data_provider` / `tests`。

## 已知局限 / 后续扩展

- **目标价/止损价触达判定未实现**：报告里有 `🎯 理想买入点` / `🛑 止损位` 等结构化价位，`outcomes.py` 的 `Outcome` 已留接口，可扩展为"窗口内是否触达目标价"。
- **"任一收盘满足即 hit"是宽松判定**：看多信号即使窗口内大跌，只要有一天涨过就算命中。可加"期末收盘 vs 基准"的严格判定做对照。
- **不分策略回测**：首版按报告整体算。`strategies/*.yaml` 定义的策略可在 `extract_signals` 阶段关联（报告里 `buy_reason` 常含策略名），做按策略命中率。
- **survivorship / 停牌**：停牌股票会被 skipped，不进分母。
- **LLM-as-judge**：报告质量的主观评估（叙述合理性、风险提示完整性）是第二阶段，需异构模型避免共模失效。

## 文件

- `extract_signals.py` - 解析 `report_*.md` → `Signal`
- `outcomes.py` - `Signal` + 行情 → `Outcome`（hit/miss/pending/skipped/no_direction，宽松+严格双口径）
- `run_eval.py` - CLI 入口 + 聚合统计（含「宽松 vs 严格对照」表）
- `recompute_strict.py` - 一次性脚本：从已落盘 jsonl 离线重算严格口径，避免重跑 akshare
