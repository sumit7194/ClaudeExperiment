#!/bin/bash
set -e
export PATH=$HOME/.local/bin:$PATH
source ~/ClaudeExperiment/scripts/checkpoint.sh

DATA_DIR="$HOME/data"
DUMPS_DIR="$DATA_DIR/dumps"

if is_done "dumpdb_2015"; then
    echo "Skipping 2015 DumpDB"
else
    echo "=== Building DumpDB for 2015 dump ==="
    time wikipedia2vec build-dump-db       "$DUMPS_DIR/enwiki-20150901-pages-articles.xml.bz2"       "$DATA_DIR/dumpdb_2015.pkl"       --pool-size 4
    mark_done "dumpdb_2015" "Done"
fi

if is_done "dumpdb_2025"; then
    echo "Skipping 2025 DumpDB"
else
    echo "=== Building DumpDB for 2025 dump ==="
    time wikipedia2vec build-dump-db       "$DUMPS_DIR/enwiki-20251220-pages-articles.xml.bz2"       "$DATA_DIR/dumpdb_2025.pkl"       --pool-size 4
    mark_done "dumpdb_2025" "Done"
fi

echo "=== DumpDB build complete ==="
ls -lh "$DATA_DIR"/dumpdb_*.pkl
