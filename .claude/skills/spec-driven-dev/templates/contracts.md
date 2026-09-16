# 股票系统易脆契约清单

> 规格驱动开发前必读。**改动涉及下列任一契约时，必须在规格的"契约影响面"中显式覆盖**，并实现后逐条回归。这些契约的历史教训：改一处、坏报告、静默。

## C1. 报告生成章节标题正则（注入脆弱点）

`src/market_analyzer.py` 的 `_CHINESE_SECTION_PATTERNS` / `_ENGLISH_SECTION_PATTERNS` 用正则定位章节标题，在标题后注入统计数据 / 指数 / 板块表格。

- **中文 (5 个)**：
  - `一、盘面总览|市场总结` → market_summary
  - `二、指数结构|指数点评|主要指数` → index_commentary
  - `三、板块主线|热点解读|板块表现` → sector_highlights
  - `四、资金与情绪|资金动向` → funds_sentiment
  - `五、消息催化|后市展望` → news_catalysts
- **英文 (3 个)**：
  - `1. Market Summary` / `Market Summary`
  - `2. Index Commentary|Major Indices`
  - `4. Sector Highlights|Sector/Theme Highlights`

**约束**：
- 新报告的章节标题**必须**能匹配对应正则：`###\s*` + 上列措辞，**不得在 `###` 后加 emoji、编号外的前缀、空格或其它符号**。
- 反例（翻车过）：LLM 输出 `### 📊 一、盘面总览` → `###\s*一、` 匹配不上 → 该表格**静默漏注入**（不报错，只少表），或板块表落到文末重复。
- 若要新增 / 替换标题措辞，需**同时**更新 `_CHINESE_SECTION_PATTERNS` 对应 pattern 的可选分支，并加回归测试覆盖带 emoji / 带前缀的负例。

## C2. 数据源优先级与降级语义

- 行情 / 板块分析：**Wind 主 / mx-data 辅**。mx-data 连调多标的会触发**分钟级限流**，部分标的两源失败→自动降级；只要 Wind 主数据完整，属于**正常降级、非错误**，不得记为分析失败。
- 单一数据源失败不应拖垮整个分析流程（除非需求明确 fail-fast）。
- 修改 `data_provider/` 时关注：优先级、失败降级、字段标准化、缓存与超时。

## C3. 报告 / 评分公式（固定值）

- 大盘 `MarketLight` 盘面信号分：`breadth(涨跌家数) × 0.45 + index(指数) × 0.35 + limit(涨跌停) × 0.20`。
- 决策仪表盘字段、买卖点位、风险警报、催化因素、操作检查清单等**报告契约为固定值**；改动需走规格并配套前后对比证据。

## C4. 补跑 / 回填数据源（真源在 DB，不在文件）

- 大盘复盘补跑时，**正确数据源是 `data/stock_analysis.db` 的 `analysis_history` 表**（`code='MARKET'`，`raw_result.raw_response` 保留原文），不要从已被覆盖的降级报告文件解析（会得到 0 指数 / 0 涨跌家数 → LLM 幻觉"接口异常"）。
- 历史日期无法回填当天新闻，`news=[]` 传入并注明"暂无可用新闻"。

## C5. 报告结构 / 通知 / Schema 变更需联动

- 改报告结构、Prompt、提取器、通知模板、机器人链路时，检查上游输入与下游消费方是否仍兼容。
- 改 API / Schema / 认证 / 报告载荷时，同时检查后端、Web、Desktop 兼容性，默认追加字段 / 保留旧字段 / 提供兼容层。
- 修改 `src/services/image_stock_extractor.py` 的 `EXTRACT_PROMPT` 时，PR 描述附**完整最新 prompt**。
- 单一通知渠道失败不应拖垮分析主流程。