#!/usr/bin/env bash
# run_quickstart.sh — Orchestrate the Wikipedia2Vec hidden-connections experiment.
#
# Quick-Start path (no training needed):
#   1. Download pre-trained embeddings + pagelinks SQL dumps
#   2. Parse link graphs
#   3. Find hidden connections (Phase 3)
#   4. Validate predictions against later dump (Phase 4) — optional
#
# Adjust the DATE variables below to match the dumps you download.
#
# Estimated runtime (quick-start, 20k entities):
#   parse_pagelinks : ~10-30 min per dump (large SQL files)
#   find_hidden     : ~5-15 min
#   validate        : ~5 min
# ----------------------------------------------------------------------------

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────

BASE_DATE="20180420"    # Date of the pre-trained model and base pagelinks dump
LATER_DATE="20250101"   # Date of the later pagelinks dump (for validation)
DATA_DIR="data"
RESULTS_DIR="results"
SCRIPTS_DIR="$(dirname "$0")"

mkdir -p "$DATA_DIR" "$RESULTS_DIR"

# ── Step 0: Download data ─────────────────────────────────────────────────────
# Uncomment the wget commands below to download the required files.
# Files are large (several GB each) — only download what you need.

echo "=== Step 0: Download data ==="
echo "Uncomment the wget lines in this script to download files."
echo ""

# Pre-trained Wikipedia2Vec model (English, 300d, ~1.4 GB compressed):
# wget -P "$DATA_DIR" \
#   "http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.pkl.bz2"

# Base pagelinks dump (for finding hidden connections):
# wget -P "$DATA_DIR" \
#   "https://dumps.wikimedia.org/enwiki/${BASE_DATE%??}/enwiki-${BASE_DATE%??}01-pagelinks.sql.gz"

# Base page dump (required to resolve page IDs to titles):
# wget -P "$DATA_DIR" \
#   "https://dumps.wikimedia.org/enwiki/${BASE_DATE%??}/enwiki-${BASE_DATE%??}01-page.sql.gz"

# Later pagelinks dump (for validation — skip if only doing predictions):
# wget -P "$DATA_DIR" \
#   "https://dumps.wikimedia.org/enwiki/${LATER_DATE%??}/enwiki-${LATER_DATE%??}01-pagelinks.sql.gz"

# Later page dump:
# wget -P "$DATA_DIR" \
#   "https://dumps.wikimedia.org/enwiki/${LATER_DATE%??}/enwiki-${LATER_DATE%??}01-page.sql.gz"

# ── Step 1: Parse link graphs ─────────────────────────────────────────────────

echo "=== Step 1: Parse base link graph (${BASE_DATE}) ==="
BASE_LINKS="$DATA_DIR/links_${BASE_DATE}.json"

if [ -f "$BASE_LINKS" ]; then
    echo "  $BASE_LINKS already exists, skipping."
else
    python "$SCRIPTS_DIR/parse_pagelinks.py" \
        --pagelinks "$DATA_DIR/enwiki-${BASE_DATE%??}01-pagelinks.sql.gz" \
        --page      "$DATA_DIR/enwiki-${BASE_DATE%??}01-page.sql.gz" \
        --output    "$BASE_LINKS"
fi

echo ""
echo "=== Step 2: Find hidden connections ==="
MODEL_FILE="$DATA_DIR/enwiki_${BASE_DATE}_300d.pkl.bz2"
CANDIDATES="$RESULTS_DIR/candidates_${BASE_DATE}.json"

if [ -f "$CANDIDATES" ]; then
    echo "  $CANDIDATES already exists, skipping."
else
    python "$SCRIPTS_DIR/find_hidden_connections.py" \
        --model           "$MODEL_FILE" \
        --links           "$BASE_LINKS" \
        --output          "$CANDIDATES" \
        --threshold       0.65 \
        --max-pairs       50000 \
        --limit-entities  20000
fi

echo ""
echo "=== Done: Phase 3 complete ==="
echo "  Candidates saved to: $CANDIDATES"
echo ""

# ── Step 3: Validate (requires later dump) ───────────────────────────────────

LATER_LINKS="$DATA_DIR/links_${LATER_DATE}.json"

if [ ! -f "$LATER_LINKS" ]; then
    echo "=== Skipping Phase 4 validation ==="
    echo "  To validate, download the ${LATER_DATE} pagelinks + page dumps,"
    echo "  then run parse_pagelinks.py to produce $LATER_LINKS"
    echo "  and re-run this script."
    exit 0
fi

echo "=== Step 3: Parse later link graph (${LATER_DATE}) ==="
python "$SCRIPTS_DIR/parse_pagelinks.py" \
    --pagelinks "$DATA_DIR/enwiki-${LATER_DATE%??}01-pagelinks.sql.gz" \
    --page      "$DATA_DIR/enwiki-${LATER_DATE%??}01-page.sql.gz" \
    --output    "$LATER_LINKS"

echo ""
echo "=== Step 4: Validate predictions ==="
python "$SCRIPTS_DIR/validate_predictions.py" \
    --candidates  "$CANDIDATES" \
    --links-later "$LATER_LINKS" \
    --links-base  "$BASE_LINKS"

echo ""
echo "=== All done! ==="
echo "  Precision@K plot: $RESULTS_DIR/precision_at_k.png"
echo "  Full results:     $RESULTS_DIR/validation_${BASE_DATE}.json"
