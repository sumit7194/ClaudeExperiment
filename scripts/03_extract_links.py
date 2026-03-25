"""Extract article-to-article links from Wikipedia DumpDB files.
Supports checkpoint/resume — saves progress every 50,000 articles."""

import json
import sys
import os
import pickle
from collections import defaultdict
from wikipedia2vec.dump_db import DumpDB
from tqdm import tqdm

SAVE_INTERVAL = 50000  # Save every 50K articles

def extract_link_graph(dump_db_path, output_path, checkpoint_path):
    """Extract all internal links with periodic checkpointing."""
    
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
    
    for i, title in enumerate(tqdm(titles[start_idx:], initial=start_idx, total=total)):
        try:
            for paragraph in dump_db.get_paragraphs(title):
                for wiki_link in paragraph.wiki_links:
                    target = wiki_link.title
                    if target:
                        link_graph[title].add(target)
        except Exception:
            continue
        
        # Periodic checkpoint
        actual_idx = start_idx + i + 1
        if actual_idx % SAVE_INTERVAL == 0:
            print(f"\n[CHECKPOINT] Saving at article {actual_idx}/{total}...")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'link_graph': link_graph, 'next_idx': actual_idx}, f)
    
    # Save final JSON output
    serializable = {k: list(v) for k, v in link_graph.items()}
    with open(output_path, 'w') as f:
        json.dump(serializable, f)
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print(f"Extracted links from {len(link_graph)} articles")
    print(f"Total unique links: {sum(len(v) for v in link_graph.values())}")
    return link_graph

if __name__ == "__main__":
    data_dir = os.path.expanduser("~/data")
    
    # Check for completion markers
    done_2015 = os.path.exists(f"{data_dir}/.checkpoints/links_2015.done")
    done_2025 = os.path.exists(f"{data_dir}/.checkpoints/links_2025.done")
    
    if not done_2015:
        print("=== Extracting 2015 link graph ===")
        extract_link_graph(
            f"{data_dir}/dumpdb_2015.pkl",
            f"{data_dir}/links_2015.json",
            f"{data_dir}/.checkpoints/links_2015.progress"
        )
        open(f"{data_dir}/.checkpoints/links_2015.done", 'w').close()
    else:
        print("Skipping 2015 links (already done)")
    
    if not done_2025:
        print("\n=== Extracting 2025 link graph ===")
        extract_link_graph(
            f"{data_dir}/dumpdb_2025.pkl",
            f"{data_dir}/links_2025.json",
            f"{data_dir}/.checkpoints/links_2025.progress"
        )
        open(f"{data_dir}/.checkpoints/links_2025.done", 'w').close()
    else:
        print("Skipping 2025 links (already done)")
