#!/usr/bin/env bash
# ================================================================
# ai-berkshire bridge — invoke deep research via Claude CLI
#
# Usage:
#   ./scripts/ai_berkshire_bridge.sh <stock_code> [stock_name] [question]
#
# Environment:
#   AI_BERKSHIRE_PATH    Path to ai-berkshire repo (default: ~/projects/ai-berkshire)
#   AI_BERKSHIRE_SKILL   Skill to invoke (default: investment-research)
#   CLAUDE_CLI           Claude CLI path (default: claude)
#
# Output:
#   Writes the research report to stdout as markdown
#   Returns 0 on success, non-zero on failure
# ================================================================

set -euo pipefail

STOCK_CODE="${1:-}"
STOCK_NAME="${2:-$STOCK_CODE}"
QUESTION="${3:-基本面综合深度分析}"

if [ -z "$STOCK_CODE" ]; then
    echo "Usage: $0 <stock_code> [stock_name] [question]" >&2
    exit 1
fi

AI_BERKSHIRE_PATH="${AI_BERKSHIRE_PATH:-$HOME/projects/ai-berkshire}"
AI_BERKSHIRE_SKILL="${AI_BERKSHIRE_SKILL:-investment-research}"
CLAUDE_CLI="${CLAUDE_CLI:-claude}"

# Resolve ai-berkshire path
if [ ! -d "$AI_BERKSHIRE_PATH" ]; then
    echo "ERROR: ai-berkshire repo not found at $AI_BERKSHIRE_PATH" >&2
    echo "Set AI_BERKSHIRE_PATH env var to the correct path" >&2
    exit 2
fi

# Verify Claude CLI is available
if ! command -v "$CLAUDE_CLI" &>/dev/null; then
    echo "ERROR: Claude CLI not found at '$CLAUDE_CLI'" >&2
    echo "Install it via: npm install -g @anthropic-ai/claude-code" >&2
    exit 2
fi

REPORT_DIR="$AI_BERKSHIRE_PATH/reports"
mkdir -p "$REPORT_DIR"

DATE_SUFFIX=$(date +%Y%m%d)
OUTPUT_FILE="$REPORT_DIR/$STOCK_NAME/$STOCK_NAME-dsa-signal-${DATE_SUFFIX}.md"
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "[ai-berkshire bridge] Running $AI_BERKSHIRE_SKILL for $STOCK_CODE ($STOCK_NAME)..." >&2
echo "[ai-berkshire bridge] Question: $QUESTION" >&2
echo "[ai-berkshire bridge] Output: $OUTPUT_FILE" >&2

# Run Claude CLI with ai-berkshire project context
# We pipe the research request via stdin and capture the output
RESEARCH_RESULT=$(cd "$AI_BERKSHIRE_PATH" && \
    "$CLAUDE_CLI" --print \
        --skill "/$AI_BERKSHIRE_SKILL" \
        --input "$STOCK_CODE $STOCK_NAME $QUESTION" \
        2>/dev/null || echo "ERROR: Claude CLI research failed for $STOCK_CODE")

# Check for error
if echo "$RESEARCH_RESULT" | grep -q "^ERROR:"; then
    echo "$RESEARCH_RESULT" >&2
    exit 3
fi

# Write the report
{
    echo "# 深度研究：$STOCK_NAME ($STOCK_CODE)"
    echo ""
    echo "**生成时间**: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "**触发来源**: daily_stock_analysis 信号驱动"
    echo "**研究技能**: ai-berkshire/$AI_BERKSHIRE_SKILL"
    echo ""
    echo "---"
    echo ""
    echo "$RESEARCH_RESULT"
} > "$OUTPUT_FILE"

echo "$OUTPUT_FILE"
exit 0
