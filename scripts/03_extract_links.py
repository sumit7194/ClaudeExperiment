"""Extract article-to-article links from Wikipedia DumpDB files.
Uses redirect resolution for both source and target titles (Iteration 1, fix #2).
Supports checkpoint/resume — saves progress every 50,000 articles."""

import json
import sys
import os
import pickle
from collections import defaultdict
from wikipedia2vec.dump_db import DumpDB
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    EXPERIMENT_CONFIG, DATA_DIR, CHECKPOINT_DIR,
    init_dirs, log_memory, set_all_seeds,
)

SAVE_INTERVAL = EXPERIMENT_CONFIG['analysis']['save_interval_articles']


def extract_link_graph(dump_db_path, output_path, checkpoint_path):
    """Extract all internal links with redirect resolution and periodic checkpointing.

    For every link found in article paragraphs:
      1. Resolve the target title through redirects so that "CRISPR/Cas9" becomes
         "CRISPR gene editing" (the canonical title).
      2. After the first pass, resolve source titles as well to collapse any
         redirected source articles into their canonical form.
    """
    # Resume from checkpoint if available
    start_idx = 0
    link_graph = defaultdict(set)

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            saved = pickle.load(f)
            link_graph = saved['link_graph']
            start_idx = saved['next_idx']
            print(f"  Loaded {len(link_graph)} articles, resuming from index {start_idx}")

    dump_db = DumpDB(dump_db_path)
    titles = list(dump_db.titles())
    total = len(titles)
    print(f"Processing {total} articles (starting from {start_idx})...")
    log_memory("Before link extraction")

    for i, title in enumerate(tqdm(titles[start_idx:], initial=start_idx, total=total)):
        try:
            for paragraph in dump_db.get_paragraphs(title):
                for wiki_link in paragraph.wiki_links:
                    target = wiki_link.title
                    if target:
                        # Resolve redirect for target title
                        resolved_target = dump_db.resolve_redirect(target)
                        link_graph[title].add(resolved_target)
        except Exception:
            continue

        # Periodic checkpoint
        actual_idx = start_idx + i + 1
        if actual_idx % SAVE_INTERVAL == 0:
            print(f"\n[CHECKPOINT] Saving at article {actual_idx}/{total}...")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'link_graph': link_graph, 'next_idx': actual_idx}, f)
            log_memory(f"Checkpoint at {actual_idx}")

    # Second pass: resolve source titles to canonical form
    print("Resolving source titles through redirects...")
    resolved_graph = defaultdict(set)
    for source, targets in tqdm(link_graph.items(), desc="Resolving sources"):
        resolved_source = dump_db.resolve_redirect(source)
        for target in targets:
            resolved_graph[resolved_source].add(target)

    # Save final JSON output
    serializable = {k: list(v) for k, v in resolved_graph.items()}
    with open(output_path, 'w') as f:
        json.dump(serializable, f)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"Extracted links from {len(resolved_graph)} articles (after redirect resolution)")
    print(f"Total unique links: {sum(len(v) for v in resolved_graph.values())}")
    log_memory("After link extraction")
    return resolved_graph


if __name__ == "__main__":
    init_dirs()
    set_all_seeds()

    # Check for completion markers
    done_2015 = os.path.exists(f"{CHECKPOINT_DIR}/links_2015.done")
    done_2025 = os.path.exists(f"{CHECKPOINT_DIR}/links_2025.done")

    if not done_2015:
        print("=== Extracting 2015 link graph (redirect-aware) ===")
        extract_link_graph(
            f"{DATA_DIR}/dumpdb_2015.pkl",
            f"{DATA_DIR}/links_2015.json",
            f"{CHECKPOINT_DIR}/links_2015.progress"
        )
        open(f"{CHECKPOINT_DIR}/links_2015.done", 'w').close()
    else:
        print("Skipping 2015 links (already done)")

    if not done_2025:
        print("\n=== Extracting 2025 link graph (redirect-aware) ===")
        extract_link_graph(
            f"{DATA_DIR}/dumpdb_2025.pkl",
            f"{DATA_DIR}/links_2025.json",
            f"{CHECKPOINT_DIR}/links_2025.progress"
        )
        open(f"{CHECKPOINT_DIR}/links_2025.done", 'w').close()
    else:
        print("Skipping 2025 links (already done)")
