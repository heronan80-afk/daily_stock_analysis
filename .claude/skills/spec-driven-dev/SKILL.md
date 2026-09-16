# Spec-Driven Dev

先写清需求与契约，再拆任务交给编码助手实现，最后对照规格验收。受开源工具 Spec Kit（规格驱动开发）启发，但不依赖其外部 CLI——以本 skill 的模板与契约清单为准。

**Repository**: https://github.com/ZhuLinsen/daily_stock_analysis

## Usage

```text
/spec-driven-dev <需求描述或场景>
# 或对已存在的规格文件反复迭代：
/spec-driven-dev <path/to/spec.md>
```

## 与其它 skill 的关系

- 立项/故障排查入口：`/analyze-issue` 判断问题真实性与边界，`/fix-issue` 落地修复
- 本 skill 负责**在动手前把"目标+验收+受影响的契约"写清楚**，再进入实现，适用信号：

  - 涉及报告结构 / 报告生成正则、数据源优先级与降级、评分公式、Schema / API / 通知载荷
  - 需求较复杂、多文件、多入口，或上次同类改动踩过契约漂移 / 验证失真的坑
  - 纯小改（typo、单变量重命名、注释）不必走规格，直接改并说明即可

## Prerequisites

先读 `templates/contracts.md`（本仓库易脆契约清单）——规格必须显式覆盖其中的相关契约，这是本 skill 的价值所在，不能跳过。

## Instructions

### Step 1: 写需求规格

用 `templates/spec.md` 起一份规格，落到 `.claude/reviews/specs/<slug>.md`（本地产物，不入库）。

规格必须写清：

1. **背景**：为什么改，解决什么问题
2. **目标 / 非目标**：明确交付边界，非目标尤其要写（防止 scope 蔓延）
3. **验收标准**：可验证、可测试的具体行为，不写宽泛描述
4. **契约影响面**：对照 `templates/contracts.md` 逐条过——本改动动了哪些"易脆契约"，等价于破坏会怎样（例如：新报告标题不满足 `###\s*一、盘面总览` → 对应表格静默漏注入）
5. **任务拆分**：把实现拆成可独立验证的小任务，标明每步的验证方式
6. **验证与回滚**：

### Step 2: 同步代码基线（如要改代码/开 PR）

按 `AGENTS.md` 拉新：

```bash
git status --short
git fetch --all --prune
# 仅当工作区干净且当前分支可 fast-forward 时执行：
git pull --ff-only
```

如存在本地改动、冲突、未跟踪风险文件、无上游或无法 fast-forward：不执行 `stash`/`reset`/强制切分支/覆盖本地状态，记录本地 HEAD、使用的远端基线与原因。

### Step 3: 交给编码助手实现

- 让实现严格对照规格的"验收标准 + 任务拆分"推进，逐任务验证，避免一次性堆大 diff
- 遵循 `AGENTS.md`：最小相关改动、复用现有模块、保持向后兼容、不顺手夹带无关重构
- 召回 `templates/contracts.md` 中被本规格标记为影响的契约，实现后逐条回归

### Step 4: 对照规格验收

- 逐条跑验收标准；涉及报告/界面改动时，按 `AGENTS.md` 附截图或前后对比
- 检查是否有**契约漂移**：同一业务语义在 runtime / API / Web / CLI / docs / tests / workflow 是否一致
- 在规格文档补一节 `[Verification]` 记录：已执行 / 未执行 / 风险 / 回滚

触发条件判定：如果实现过程暴露了新契约或发现原验收标准不成立，回到 Step 1 修订规格后再继续，而不是堆叠补丁。

### Step 5: 合入前检查

- 涉及用户可见能力、报告结构、Schema / API / 通知、部署、调度时，同步更新 `docs/CHANGELOG.md`（`[Unreleased]` 扁平格式）、相关 `docs/*.md`、`.env.example`
- `check_ai_assets.py` 覆盖的资产若未动，则不重复执行；动了 `.claude/skills/` 相关才需要跑
- 只有用户明确确认后才：建分支 / `git commit` / `git push` / 创建 PR

## Allowed Auto-Actions (No Confirmation Needed)

- 阅读代码与 `templates/contracts.md`
- 写/改本地规格文档（`.claude/reviews/specs/*.md`）
- 执行 `git fetch --all --prune`，并在工作区干净且可 fast-forward 时 `git pull --ff-only`
- 运行非破坏性的本地验证

## Actions Requiring Confirmation

1. 切换或创建分支
2. `git commit`
3. `git push`
4. 创建 PR