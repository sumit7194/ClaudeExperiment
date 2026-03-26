#!/bin/bash
# Master pipeline runner for Temporal Knowledge Gap Finder
# Fully checkpoint-aware — safe to restart at any point
# Memory-safe: each Python step loads/frees resources in phases
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

# Step 3: Extract redirect-aware link graphs
# Memory phase: loads one DumpDB at a time, frees before loading the next
echo "$LOG_PREFIX === STEP 3: Extracting redirect-aware link graphs ==="
python3 "$SCRIPTS_DIR/03_extract_links.py"
echo "$LOG_PREFIX Step 3 complete at $(date)"

# Step 4: Find hidden connections (stratified sampling, threshold-free, deduplication)
# Memory phase A: loads 2015 model + links, finds candidates, frees
# Memory phase B: loads 2025 model + links, finds candidates, frees
echo "$LOG_PREFIX === STEP 4: Finding hidden connections (stratified + threshold-free) ==="
python3 "$SCRIPTS_DIR/04_find_hidden_connections.py"
echo "$LOG_PREFIX Step 4 complete at $(date)"

# Step 5: Validate predictions with full analysis
# Memory phase 1-3: loads link sets, runs stratified validation + multi-threshold
# Memory phase 4-5: statistical significance + random baseline
# Memory phase 6:   loads link graph for BFS shortest paths, then frees
# Memory phase 7:   generates HTML report
echo "$LOG_PREFIX === STEP 5: Validating predictions (stratified + significance + BFS + report) ==="
python3 "$SCRIPTS_DIR/05_validate.py"
echo "$LOG_PREFIX Step 5 complete at $(date)"

# Step 6: Visualizations and real-world applications
# Memory phase: loads results JSONs (lightweight), generates plots + reports
echo "$LOG_PREFIX === STEP 6: Generating visualizations and applications ==="
python3 "$SCRIPTS_DIR/06_visualize.py"
echo "$LOG_PREFIX Step 6 complete at $(date)"

# Commit results to git
echo "$LOG_PREFIX === Pushing results to GitHub ==="
cd "$HOME/ClaudeExperiment"
mkdir -p results
cp "$HOME/data/results"/*.json ./results/ 2>/dev/null || true
cp "$HOME/data/results"/*.html ./results/ 2>/dev/null || true
cp "$HOME/data/results"/*.png ./results/ 2>/dev/null || true
cp "$HOME/data/results"/*.md ./results/ 2>/dev/null || true
git add scripts/ results/
git commit -m "Pipeline complete: add scripts and experiment results" || true
git push origin main || echo "Push failed — will retry manually"

echo "$LOG_PREFIX === PIPELINE COMPLETE at $(date) ==="
show_checkpoints
