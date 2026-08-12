#!/usr/bin/env bash
# ================================================================
# Wrapper: daily stock analysis + ai-berkshire deep research
#
# Runs the standard daily analysis, then detects signal stocks and
# invokes ai-berkshire deep research on them.
#
# Usage:
#   ./scripts/run_with_berkshire.sh [--schedule|--full|--stocks-only]
#
# Environment:
#   AI_BERKSHIRE_ENABLED=true  (default: false)
#   AI_BERKSHIRE_PATH          (default: ~/projects/ai-berkshire)
# ================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

# Parse args
MODE="${1:-}"
SCHEDULE_FLAG=""
FORCE_RUN_FLAG=""

if [ "$MODE" = "--schedule" ]; then
    SCHEDULE_FLAG="--schedule"
elif [ "$MODE" = "--force-run" ]; then
    FORCE_RUN_FLAG="--force-run"
fi

echo "=========================================="
echo "🚀 Daily Stock Analysis + AI Berkshire"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Step 1: Run the daily stock analysis
echo ""
echo "📊 Step 1/2: Running daily stock analysis..."
echo "=========================================="
if [ -n "$SCHEDULE_FLAG" ]; then
    python main.py --schedule
else
    python main.py $FORCE_RUN_FLAG
fi
ANALYSIS_EXIT=$?

echo ""
echo "=========================================="
echo "✅ Daily analysis completed (exit=$ANALYSIS_EXIT)"
echo "=========================================="

# Step 2: Run ai-berkshire bridge (if enabled and analysis succeeded)
AI_BERKSHIRE_ENABLED="${AI_BERKSHIRE_ENABLED:-true}"
if [ "$AI_BERKSHIRE_ENABLED" = "true" ] && [ "$ANALYSIS_EXIT" -eq 0 ]; then
    echo ""
    echo "🔬 Step 2/2: Running ai-berkshire deep research bridge..."
    echo "=========================================="

    AI_BERKSHIRE_PATH="${AI_BERKSHIRE_PATH:-$HOME/projects/ai-berkshire}"
    AI_BERKSHIRE_SKILL="${AI_BERKSHIRE_SKILL:-investment-research}"
    BRIDGE_SIGNAL_MIN="${BRIDGE_SIGNAL_MIN:-70}"

    # Ensure ai-berkshire Claude commands are installed
    if [ -f "$AI_BERKSHIRE_PATH/scripts/install-claude-commands.sh" ]; then
        bash "$AI_BERKSHIRE_PATH/scripts/install-claude-commands.sh" 2>/dev/null || true
    fi

    # Run the bridge — detect signals from the latest reports
    python scripts/ai_berkshire_bridge.py \
        --results reports/ \
        --min-score "$BRIDGE_SIGNAL_MIN" \
        --skill "$AI_BERKSHIRE_SKILL" \
        --ai-berkshire-path "$AI_BERKSHIRE_PATH" \
        2>&1

    BRIDGE_EXIT=$?
    echo ""
    echo "=========================================="
    if [ "$BRIDGE_EXIT" -eq 0 ]; then
        echo "✅ AI Berkshire bridge completed"
    else
        echo "⚠️ AI Berkshire bridge completed with warnings (exit=$BRIDGE_EXIT)"
    fi
    echo "=========================================="
elif [ "$AI_BERKSHIRE_ENABLED" != "true" ]; then
    echo ""
    echo "⏭️  AI Berkshire deep research disabled (AI_BERKSHIRE_ENABLED != true)"
    echo "   To enable: export AI_BERKSHIRE_ENABLED=true"
fi

echo ""
echo "🎯 All done at $(date '+%Y-%m-%d %H:%M:%S')"
