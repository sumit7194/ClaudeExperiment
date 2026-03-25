#!/bin/bash
# Checkpoint utility functions for pipeline resilience
# Source this in other scripts: source ~/ClaudeExperiment/scripts/checkpoint.sh

CHECKPOINT_DIR="$HOME/data/.checkpoints"
mkdir -p "$CHECKPOINT_DIR"

mark_done() {
    local step="$1"
    local info="$2"
    echo "$(date -Iseconds) | $info" > "$CHECKPOINT_DIR/$step.done"
    echo "[CHECKPOINT] Marked '$step' as done"
}

is_done() {
    local step="$1"
    if [ -f "$CHECKPOINT_DIR/$step.done" ]; then
        echo "[CHECKPOINT] '$step' already completed ($(cat "$CHECKPOINT_DIR/$step.done"))"
        return 0
    fi
    return 1
}

clear_checkpoint() {
    local step="$1"
    rm -f "$CHECKPOINT_DIR/$step.done"
    echo "[CHECKPOINT] Cleared '$step'"
}

show_checkpoints() {
    echo "=== Pipeline Checkpoints ==="
    for f in "$CHECKPOINT_DIR"/*.done 2>/dev/null; do
        [ -f "$f" ] || continue
        step=$(basename "$f" .done)
        info=$(cat "$f")
        printf "  %-30s %s\n" "$step" "$info"
    done
}
