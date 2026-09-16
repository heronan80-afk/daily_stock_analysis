# Repository Claude Skills

本目录存放仓库级协作 skills，属于版本库资产。

- 规则真源：仓库根目录 `AGENTS.md`
- 兼容入口：根目录 `CLAUDE.md`（应为指向 `AGENTS.md` 的软链接）
- 本目录中的 skill 需要与 `AGENTS.md` 保持一致
- `.claude/reviews/` 属于本地分析产物，不作为规则真源

如果未来需要兼容其他 agent 目录（如 `.agents/skills/` 或 `.github/skills/`），应先明确单一真源，再通过脚本或镜像同步，而不是手工长期维护多份同义内容。

## 已注册 skills

| skill | 用途 | 调用 |
| --- | --- | --- |
| `analyze-issue` | 分析 GitHub Issue 的真实性、优先级、边界与建议动作 | `/analyze-issue <issue_number>` |
| `analyze-pr` | 审查 PR 的各维度证据与合入判定 | `/analyze-pr <pr_number>` |
| `fix-issue` | 基于 issue 分析落地修复并补齐验证/风险/回滚 | `/fix-issue <issue_number>` |
| `spec-driven-dev` | 规格驱动开发：动手前写清需求+验收+契约影响面，再拆任务实现追踪 | `/spec-driven-dev <需求>`（详见其 `templates/contracts.md`） |
