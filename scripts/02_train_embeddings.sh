#!/bin/bash
# Step 2: Train Wikipedia2Vec embeddings using individual subcommands
# Each substep is checkpointed so we can resume after failures
set -e
export PATH=$HOME/.local/bin:$PATH
source ~/ClaudeExperiment/scripts/checkpoint.sh

DATA_DIR="$HOME/data"
MODELS_DIR="$DATA_DIR/models"
mkdir -p "$MODELS_DIR"

train_year() {
    local YEAR=$1
    local DUMPDB="$DATA_DIR/dumpdb_${YEAR}.pkl"
    local DICT="$DATA_DIR/dict_${YEAR}.pkl"
    local LINK_GRAPH="$DATA_DIR/link_graph_${YEAR}.pkl"
    local MENTION_DB="$DATA_DIR/mention_db_${YEAR}.pkl"
    local MODEL="$MODELS_DIR/wiki${YEAR}.pkl"

    echo "============================================"
    echo "  Training $YEAR embeddings"
    echo "============================================"

    # Sub-step 2a: Build dictionary
    if is_done "dict_${YEAR}"; then
        echo "Skipping dictionary for $YEAR"
    else
        echo "--- Building dictionary for $YEAR ---"
        time wikipedia2vec build-dictionary \
            "$DUMPDB" "$DICT" \
            --min-entity-count 5 \
            --lowercase \
            --pool-size 4
        mark_done "dict_${YEAR}" "Size: $(du -h "$DICT" | cut -f1)"
    fi

    # Sub-step 2b: Build link graph
    if is_done "link_graph_${YEAR}"; then
        echo "Skipping link graph for $YEAR"
    else
        echo "--- Building link graph for $YEAR ---"
        time wikipedia2vec build-link-graph \
            "$DUMPDB" "$DICT" "$LINK_GRAPH" \
            --pool-size 4
        mark_done "link_graph_${YEAR}" "Size: $(du -h "$LINK_GRAPH" | cut -f1)"
    fi

    # Sub-step 2c: Build mention DB
    if is_done "mention_db_${YEAR}"; then
        echo "Skipping mention DB for $YEAR"
    else
        echo "--- Building mention DB for $YEAR ---"
        time wikipedia2vec build-mention-db \
            "$DUMPDB" "$DICT" "$MENTION_DB" \
            --pool-size 4
        mark_done "mention_db_${YEAR}" "Size: $(du -h "$MENTION_DB" | cut -f1)"
    fi

    # Sub-step 2d: Train embeddings (the long one)
    if is_done "model_${YEAR}"; then
        echo "Skipping embedding training for $YEAR"
    else
        echo "--- Training embeddings for $YEAR (this takes hours) ---"
        time wikipedia2vec train-embedding \
            "$DUMPDB" "$DICT" "$MODEL" \
            --link-graph "$LINK_GRAPH" \
            --mention-db "$MENTION_DB" \
            --dim-size 300 \
            --window 10 \
            --iteration 10 \
            --negative 15 \
            --pool-size 4
        mark_done "model_${YEAR}" "Size: $(du -h "$MODEL" | cut -f1)"
    fi

    echo "=== $YEAR training pipeline complete ==="
}

# Train both years
train_year "2015"
train_year "2025"

echo "=== All training complete ==="
ls -lh "$MODELS_DIR"/*.pkl
show_checkpoints
