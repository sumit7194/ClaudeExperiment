"""
find_hidden_connections.py — Phase 3: Find entity pairs close in embedding space
but NOT linked in Wikipedia. These are "hidden connection" candidates.

Usage:
    python scripts/find_hidden_connections.py \\
        --model   data/enwiki_20180420_300d.pkl.bz2 \\
        --links   data/links_2018.json \\
        --output  results/candidates_2018.json \\
        [--threshold 0.65] \\
        [--max-pairs 50000] \\
        [--limit-entities 20000]

Pre-trained models available at:
    https://wikipedia2vec.github.io/wikipedia2vec/pretrained/
"""

import argparse
import json
import os
import sys

from tqdm import tqdm
from wikipedia2vec import Wikipedia2Vec

# parse_pagelinks.py lives in the same directory
sys.path.insert(0, os.path.dirname(__file__))
from parse_pagelinks import load_link_set


def find_close_but_unlinked(model, link_set,
                             similarity_threshold=0.65,
                             max_pairs=50000,
                             limit_entities=20000,
                             min_entity_name_length=3):
    """
    Find entity pairs that are close in embedding space but NOT linked in Wikipedia.

    For each entity, fetch its nearest neighbors in embedding space. Any neighbor
    that passes the similarity threshold but has no Wikipedia link to the source
    entity is a "hidden connection" candidate.

    Args:
        model: Wikipedia2Vec model instance.
        link_set: set of (title_lower, title_lower) pairs (bidirectional).
        similarity_threshold: Minimum cosine similarity to consider a pair.
        max_pairs: Stop after collecting this many candidates.
        limit_entities: Only scan the first N entities (for tractability).
        min_entity_name_length: Skip very short entity names (noise filter).

    Returns:
        list of dicts sorted by similarity descending:
            {'entity_a': str, 'entity_b': str, 'similarity': float}
    """
    entities = [
        e for e in model.dictionary.entities()
        if len(e.title) >= min_entity_name_length
    ]

    if limit_entities and limit_entities < len(entities):
        entities = entities[:limit_entities]

    print(f"Scanning {len(entities):,} entities for hidden connections ...")
    print(f"  similarity_threshold={similarity_threshold}, max_pairs={max_pairs}")

    seen_pairs = set()
    candidates = []

    for entity in tqdm(entities):
        if len(candidates) >= max_pairs:
            break
        try:
            neighbors = model.most_similar(entity, count=50)
        except Exception:
            continue

        for neighbor_item, similarity in neighbors:
            if similarity < similarity_threshold:
                break  # results are sorted descending

            # We want entity–entity pairs only (not word results)
            if not hasattr(neighbor_item, "title"):
                continue

            neighbor = neighbor_item
            a_title = entity.title
            b_title = neighbor.title

            # Deduplicate symmetric pairs
            pair_key = tuple(sorted([a_title.lower(), b_title.lower()]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            a_lower = a_title.lower()
            b_lower = b_title.lower()

            # The key check: close in embedding space but NOT linked
            if (a_lower, b_lower) not in link_set and (b_lower, a_lower) not in link_set:
                candidates.append({
                    "entity_a": a_title,
                    "entity_b": b_title,
                    "similarity": float(similarity),
                })

                if len(candidates) >= max_pairs:
                    break

    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Path to Wikipedia2Vec .pkl or .pkl.bz2 file")
    parser.add_argument("--links", required=True, help="Path to title-keyed links JSON (from parse_pagelinks.py)")
    parser.add_argument("--output", required=True, help="Output JSON path, e.g. results/candidates_2018.json")
    parser.add_argument("--threshold", type=float, default=0.65, help="Min cosine similarity (default: 0.65)")
    parser.add_argument("--max-pairs", type=int, default=50000, help="Max candidate pairs to collect (default: 50000)")
    parser.add_argument("--limit-entities", type=int, default=20000,
                        help="Scan only the first N entities for speed (default: 20000, 0=all)")
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    model = Wikipedia2Vec.load(args.model)
    print(f"  {len(list(model.dictionary.entities())):,} entities, "
          f"{len(list(model.dictionary.words())):,} words")

    print(f"Loading link set from {args.links} ...")
    link_set = load_link_set(args.links)
    print(f"  {len(link_set):,} directed link pairs loaded")

    limit = args.limit_entities if args.limit_entities > 0 else None
    candidates = find_close_but_unlinked(
        model,
        link_set,
        similarity_threshold=args.threshold,
        max_pairs=args.max_pairs,
        limit_entities=limit,
    )

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(candidates, f, indent=2)

    print(f"\nFound {len(candidates):,} hidden-connection candidates.")
    print(f"Saved to {args.output}")

    print(f"\nTop 20 predictions:")
    for c in candidates[:20]:
        print(f"  {c['entity_a']!r:40s} <-> {c['entity_b']!r:40s}  sim={c['similarity']:.3f}")


if __name__ == "__main__":
    main()
