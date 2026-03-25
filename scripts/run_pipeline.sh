#!/bin/bash
# Master pipeline runner for Temporal Knowledge Gap Finder
# Fully checkpoint-aware — safe to restart at any point
# Run with: nohup bash ~/ClaudeExperiment/scripts/run_pipeline.sh > ~/pipeline.log 2>&1 &

set -e
export PATH=$HOME/.local/bin:$PATH
source ~/ClaudeExperiment/scripts/checkpoint.sh

SCRIPTS_DIR="$HOME/ClaudeExperiment/scripts"
LOG_PREFIX="[PIPELINE]"

echo "$LOG_PREFIX Starting pipeline at $(date)"
echo "$LOG_PREFIX Checking existing checkpoints..."
show_checkpoints

# Step 1: Build DumpDBs
echo "$LOG_PREFIX === STEP 1: Building DumpDBs ==="
bash "$SCRIPTS_DIR/01_build_dumpdb.sh"
echo "$LOG_PREFIX Step 1 complete at $(date)"

# Step 2: Train embeddings (individual subcommands with checkpoints)
echo "$LOG_PREFIX === STEP 2: Training embeddings ==="
bash "$SCRIPTS_DIR/02_train_embeddings.sh"
echo "$LOG_PREFIX Step 2 complete at $(date)"

# Step 3: Extract link graphs
echo "$LOG_PREFIX === STEP 3: Extracting link graphs ==="
python3 "$SCRIPTS_DIR/03_extract_links.py"
echo "$LOG_PREFIX Step 3 complete at $(date)"

# Step 4: Find hidden connections
echo "$LOG_PREFIX === STEP 4: Finding hidden connections ==="
python3 "$SCRIPTS_DIR/04_find_hidden_connections.py"
echo "$LOG_PREFIX Step 4 complete at $(date)"

# Step 5: Validate predictions
echo "$LOG_PREFIX === STEP 5: Validating predictions ==="
python3 "$SCRIPTS_DIR/05_validate.py"
echo "$LOG_PREFIX Step 5 complete at $(date)"

# Commit results to git
echo "$LOG_PREFIX === Pushing results to GitHub ==="
cd "$HOME/ClaudeExperiment"
mkdir -p results
cp "$HOME/data/results"/*.json ./results/ 2>/dev/null || true
git add scripts/ skills/ results/
git commit -m "Pipeline complete: add scripts, skill, and experiment results" || true
git push origin main || echo "Push failed — will retry manually"

echo "$LOG_PREFIX === PIPELINE COMPLETE at $(date) ==="
show_checkpoints
