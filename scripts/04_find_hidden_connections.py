"""Find concept pairs close in embedding space but NOT linked.
Improvements applied:
  - Stratified frequency sampling (Iteration 1, fix #1)
  - Threshold-free candidate generation (Iteration 1, fix #3)
  - Symmetric pair deduplication (Iteration 2)
  - Memory usage logging
  - Reproducible random seeds from config.py
Saves progress every 1000 entities — fully resumable."""

import gc
import json
import os
import pickle
import random
import sys

import numpy as np
from wikipedia2vec import Wikipedia2Vec
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    EXPERIMENT_CONFIG, DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR,
    init_dirs, set_all_seeds, log_memory,
)

SAVE_INTERVAL = EXPERIMENT_CONFIG['analysis']['save_interval_entities']
ANALYSIS = EXPERIMENT_CONFIG['analysis']


# ---------------------------------------------------------------------------
# Link set helpers
# ---------------------------------------------------------------------------

def load_link_set(links_path):
    """Load link graph as a set of (source, target) pairs (both directions)."""
    with open(links_path) as f:
        link_dict = json.load(f)
    link_set = set()
    for source, targets in link_dict.items():
        for target in targets:
            link_set.add((source.lower(), target.lower()))
            link_set.add((target.lower(), source.lower()))
    return link_set


def are_linked(entity_a, entity_b, link_set):
    """Check if two entities are linked (either direction)."""
    a = entity_a.title.lower()
    b = entity_b.title.lower()
    return (a, b) in link_set or (b, a) in link_set


# ---------------------------------------------------------------------------
# Stratified sampling (Iteration 1, fix #1)
# ---------------------------------------------------------------------------

def stratified_sample(model, total_n=None):
    """Sample entities across frequency tiers for representative coverage.

    Instead of the old entities[:20000], this sorts all entities by their
    reference count and draws equally from five frequency tiers so that
    high-freq, mid-freq, and long-tail entities are all represented.
    """
    if total_n is None:
        total_n = ANALYSIS['sample_size']

    min_name_len = ANALYSIS['min_entity_name_length']

    all_entities = [e for e in model.dictionary.entities()
                    if len(e.title) >= min_name_len]
    all_entities.sort(key=lambda e: e.count, reverse=True)

    n = len(all_entities)
    tiers = {
        'high_freq':  (0, int(n * 0.01)),
        'upper_mid':  (int(n * 0.01), int(n * 0.05)),
        'mid_freq':   (int(n * 0.05), int(n * 0.20)),
        'lower_mid':  (int(n * 0.20), int(n * 0.50)),
        'long_tail':  (int(n * 0.50), n),
    }

    samples_per_tier = total_n // len(tiers)
    sampled = []
    tier_info = {}

    for tier_name, (start, end) in tiers.items():
        tier_entities = all_entities[start:end]
        k = min(samples_per_tier, len(tier_entities))
        chosen = random.sample(tier_entities, k)
        sampled.extend(chosen)
        tier_info[tier_name] = {
            'range': f'{start}-{end}',
            'tier_size': len(tier_entities),
            'sampled': k,
        }

    print(f"Stratified sample: {len(sampled)} entities from {n} total")
    for name, info in tier_info.items():
        print(f"  {name}: {info['sampled']} from {info['tier_size']} "
              f"(range {info['range']})")

    return sampled, tier_info


# ---------------------------------------------------------------------------
# Deduplication (Iteration 2)
# ---------------------------------------------------------------------------

def deduplicate_candidates(candidates):
    """Remove duplicate / symmetric pairs. Keep the entry with higher similarity."""
    seen = {}
    for c in candidates:
        key = tuple(sorted([c['entity_a'].lower(), c['entity_b'].lower()]))
        if key not in seen or c['similarity'] > seen[key]['similarity']:
            seen[key] = c

    deduped = sorted(seen.values(), key=lambda x: x['similarity'], reverse=True)

    removed = len(candidates) - len(deduped)
    pct = (removed / len(candidates) * 100) if candidates else 0
    print(f"Deduplication: {len(candidates)} -> {len(deduped)} "
          f"({removed} duplicates removed, {pct:.1f}%)")
    return deduped


# ---------------------------------------------------------------------------
# Core candidate finder — threshold-free (Iteration 1, fix #3)
# ---------------------------------------------------------------------------

def find_close_but_unlinked(model, link_set, entities, output_path, checkpoint_path,
                            top_k_neighbors=None):
    """Find hidden connections with periodic checkpointing.

    Unlike the original version, this does NOT apply a fixed similarity
    threshold.  All unlinked neighbors (up to top_k per entity) are
    collected; the multi-threshold precision analysis in 05_validate.py
    determines which similarity range is actually predictive.
    """
    if top_k_neighbors is None:
        top_k_neighbors = ANALYSIS['top_k_neighbors']

    # Resume from checkpoint
    start_idx = 0
    candidates = []
    seen_pairs = set()

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            saved = pickle.load(f)
            candidates = saved['candidates']
            seen_pairs = saved['seen_pairs']
            start_idx = saved['next_idx']
            print(f"  Loaded {len(candidates)} candidates, "
                  f"resuming from entity {start_idx}")

    print(f"Analyzing {len(entities)} entities (starting from {start_idx}), "
          f"top_k_neighbors={top_k_neighbors}, NO similarity threshold...")
    log_memory("Before candidate search")

    for i, entity in enumerate(tqdm(entities[start_idx:],
                                    initial=start_idx,
                                    total=len(entities))):
        try:
            neighbors = model.most_similar(entity, count=top_k_neighbors)
            for neighbor_item, similarity in neighbors:
                if not hasattr(neighbor_item, 'title'):
                    continue

                # Deduplicate symmetric pairs on the fly
                pair_key = tuple(sorted([entity.title.lower(),
                                         neighbor_item.title.lower()]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                if not are_linked(entity, neighbor_item, link_set):
                    candidates.append({
                        'entity_a': entity.title,
                        'entity_b': neighbor_item.title,
                        'similarity': float(similarity),
                        'entity_a_count': entity.count,
                        'entity_b_count': neighbor_item.count,
                    })
        except Exception:
            continue

        # Periodic checkpoint
        actual_idx = start_idx + i + 1
        if actual_idx % SAVE_INTERVAL == 0:
            print(f"\n[CHECKPOINT] entity {actual_idx}/{len(entities)}, "
                  f"{len(candidates)} candidates so far")
            log_memory(f"Checkpoint at {actual_idx}")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'candidates': candidates,
                    'seen_pairs': seen_pairs,
                    'next_idx': actual_idx,
                }, f)

    candidates.sort(key=lambda x: x['similarity'], reverse=True)

    # Final deduplication pass (catches any remaining symmetric pairs)
    candidates = deduplicate_candidates(candidates)

    # Save final output
    with open(output_path, 'w') as f:
        json.dump(candidates, f, indent=2)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    log_memory("After candidate search")
    return candidates


# ---------------------------------------------------------------------------
# Main — memory-safe phased execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_dirs()
    set_all_seeds()

    # ── PHASE A: 2015 candidates ──
    done_2015 = os.path.join(CHECKPOINT_DIR, "hidden_2015.done")
    if not os.path.exists(done_2015):
        print("=== Phase A: 2015 hidden connection search ===")
        log_memory("Loading 2015 model")
        model_2015 = Wikipedia2Vec.load(f"{DATA_DIR}/models/wiki2015.pkl")
        log_memory("Loading 2015 links")
        links_2015 = load_link_set(f"{DATA_DIR}/links_2015.json")

        sampled_2015, tier_info_2015 = stratified_sample(model_2015)

        # Save sampling info
        with open(f"{RESULTS_DIR}/sampling_info_2015.json", 'w') as f:
            json.dump(tier_info_2015, f, indent=2)

        candidates_2015 = find_close_but_unlinked(
            model_2015, links_2015, sampled_2015,
            f"{RESULTS_DIR}/candidates_2015.json",
            f"{CHECKPOINT_DIR}/hidden_2015.progress",
        )
        print(f"\nFound {len(candidates_2015)} candidate hidden connections (2015)")
        print("\nTop 20 predictions:")
        for c in candidates_2015[:20]:
            print(f"  {c['entity_a']} <--> {c['entity_b']}  "
                  f"(sim: {c['similarity']:.3f})")

        open(done_2015, 'w').close()

        # Free memory before next phase
        del model_2015, links_2015, sampled_2015, candidates_2015
        gc.collect()
        log_memory("After freeing 2015 resources")
    else:
        print("Skipping 2015 analysis (already done)")

    # ── PHASE B: 2025 candidates (future predictions) ──
    done_2025 = os.path.join(CHECKPOINT_DIR, "hidden_2025.done")
    if not os.path.exists(done_2025):
        print("\n=== Phase B: 2025 hidden connection search ===")
        log_memory("Loading 2025 model")
        model_2025 = Wikipedia2Vec.load(f"{DATA_DIR}/models/wiki2025.pkl")
        log_memory("Loading 2025 links")
        links_2025 = load_link_set(f"{DATA_DIR}/links_2025.json")

        sampled_2025, tier_info_2025 = stratified_sample(model_2025)

        with open(f"{RESULTS_DIR}/sampling_info_2025.json", 'w') as f:
            json.dump(tier_info_2025, f, indent=2)

        candidates_2025 = find_close_but_unlinked(
            model_2025, links_2025, sampled_2025,
            f"{RESULTS_DIR}/candidates_2025.json",
            f"{CHECKPOINT_DIR}/hidden_2025.progress",
        )
        print(f"\nFound {len(candidates_2025)} candidate hidden connections (2025/future)")
        print("\nTop 20 predictions:")
        for c in candidates_2025[:20]:
            print(f"  {c['entity_a']} <--> {c['entity_b']}  "
                  f"(sim: {c['similarity']:.3f})")

        open(done_2025, 'w').close()

        del model_2025, links_2025, sampled_2025, candidates_2025
        gc.collect()
        log_memory("After freeing 2025 resources")
    else:
        print("Skipping 2025 analysis (already done)")
