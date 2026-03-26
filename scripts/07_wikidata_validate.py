#!/usr/bin/env python3
"""
Step 7: Wikidata Cross-Validation
Validates embedding predictions against Wikidata's independent knowledge graph.
Uses SPARQL endpoint — no download needed.
"""

import json
import os
import sys
import time
import random
from collections import Counter
from datetime import datetime

import requests

sys.path.insert(0, os.path.dirname(__file__))
from config import EXPERIMENT_CONFIG, set_all_seeds, log_memory

set_all_seeds()

DATA_DIR = EXPERIMENT_CONFIG['directories']['data']
RESULTS_DIR = EXPERIMENT_CONFIG['directories']['results']

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "TemporalKnowledgeGapFinder/1.0 (research experiment; contact: github.com/sumit7194/ClaudeExperiment)"


def query_wikidata_relations(entity_a, entity_b, max_retries=3):
    """
    Check if two entities have ANY Wikidata relationship.
    Uses the free SPARQL endpoint.
    """
    # Escape quotes in entity names
    a_clean = entity_a.replace('"', '\\"')
    b_clean = entity_b.replace('"', '\\"')

    sparql = f"""
    SELECT ?prop ?propLabel WHERE {{
      ?item1 rdfs:label "{a_clean}"@en .
      ?item2 rdfs:label "{b_clean}"@en .
      {{ ?item1 ?prop ?item2 . }} UNION {{ ?item2 ?prop ?item1 . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 10
    """

    for attempt in range(max_retries):
        try:
            response = requests.get(
                SPARQL_ENDPOINT,
                params={'query': sparql, 'format': 'json'},
                headers={'User-Agent': USER_AGENT},
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', {}).get('bindings', [])
                relations = []
                for r in results:
                    label = r.get('propLabel', {}).get('value', 'unknown')
                    relations.append(label)
                return relations
            elif response.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                time.sleep(5)
        except Exception as e:
            time.sleep(5)

    return []


def batch_wikidata_validation(candidates, top_n=200):
    """
    Validate top predictions against Wikidata.
    Limited to top_n to respect SPARQL rate limits.
    """
    print(f"\nValidating top {top_n} predictions against Wikidata...")
    results = []
    has_relation = 0

    for i, c in enumerate(candidates[:top_n]):
        if i % 25 == 0:
            print(f"  Progress: {i}/{top_n} ({has_relation} with Wikidata relations)")

        rels = query_wikidata_relations(c['entity_a'], c['entity_b'])
        c_copy = c.copy()
        c_copy['wikidata_relations'] = rels
        c_copy['has_wikidata_link'] = len(rels) > 0
        results.append(c_copy)

        if rels:
            has_relation += 1

        time.sleep(1.5)  # Be a good API citizen

    return results, has_relation


def random_wikidata_baseline(all_entity_titles, n=100):
    """
    Check random entity pairs for Wikidata relations (baseline).
    """
    print(f"\nComputing random Wikidata baseline ({n} pairs)...")
    has_relation = 0
    titles = list(all_entity_titles)

    for i in range(n):
        if i % 25 == 0:
            print(f"  Random baseline: {i}/{n}")
        a = random.choice(titles)
        b = random.choice(titles)
        if a == b:
            continue
        rels = query_wikidata_relations(a, b)
        if rels:
            has_relation += 1
        time.sleep(1.5)

    return has_relation / n


def main():
    print("=" * 60)
    print("  STEP 7: WIKIDATA CROSS-VALIDATION")
    print("=" * 60)
    log_memory("start")

    # Load candidates
    candidates_path = f"{RESULTS_DIR}/candidates_2015.json"
    if not os.path.exists(candidates_path):
        print("No candidates found. Run step 4 first.")
        return

    with open(candidates_path) as f:
        candidates = json.load(f)

    # Sort by similarity (highest first)
    candidates.sort(key=lambda x: x['similarity'], reverse=True)

    # Validate top predictions
    top_n = min(200, len(candidates))
    validated, n_with_relations = batch_wikidata_validation(candidates, top_n=top_n)

    wikidata_precision = n_with_relations / top_n

    # Random baseline (smaller sample due to rate limits)
    all_titles = set()
    for c in candidates:
        all_titles.add(c['entity_a'])
        all_titles.add(c['entity_b'])

    random_precision = random_wikidata_baseline(list(all_titles), n=50)

    # Collect relation types
    relation_types = Counter()
    for v in validated:
        for rel in v.get('wikidata_relations', []):
            relation_types[rel] += 1

    # Results
    ratio = wikidata_precision / max(random_precision, 1e-6)

    print(f"\n{'=' * 60}")
    print(f"  WIKIDATA CROSS-VALIDATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Embedding top-{top_n} with Wikidata relation: {wikidata_precision:.3f} ({n_with_relations}/{top_n})")
    print(f"  Random pairs with Wikidata relation:          {random_precision:.3f}")
    print(f"  Ratio:                                        {ratio:.1f}x")
    print(f"")
    if relation_types:
        print(f"  Most common Wikidata relations in predictions:")
        for rel, count in relation_types.most_common(10):
            print(f"    {rel}: {count}")

    # Save results
    results = {
        'top_n': top_n,
        'wikidata_precision': wikidata_precision,
        'n_with_relations': n_with_relations,
        'random_baseline': random_precision,
        'ratio': ratio,
        'top_relation_types': dict(relation_types.most_common(20)),
        'timestamp': datetime.now().isoformat(),
    }

    output_path = f"{RESULTS_DIR}/wikidata_validation.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Also save detailed results with per-pair relations
    detailed_path = f"{RESULTS_DIR}/wikidata_validation_detailed.json"
    with open(detailed_path, 'w') as f:
        json.dump(validated, f, indent=2)

    log_memory("end")
    print(f"\n{'='*60}")
    if ratio > 2:
        print("  ✅ Predictions validated against independent knowledge graph!")
    elif ratio > 1.2:
        print("  ⚠️  Weak signal in Wikidata cross-validation")
    else:
        print("  ❌ No signal in Wikidata cross-validation")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
