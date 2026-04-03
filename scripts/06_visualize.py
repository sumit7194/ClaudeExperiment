#!/usr/bin/env python3
"""
Step 6: Visualization and Real-World Applications
Generates publication-ready visualizations and practical outputs.
Memory-safe: loads one model at a time.
"""

import json
import os
import gc
import random
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import EXPERIMENT_CONFIG, DATA_DIR, RESULTS_DIR, set_all_seeds, log_memory

set_all_seeds()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════
#  VISUALIZATION 1: Confidence Ladder
# ═══════════════════════════════════════════════════════════

def _pair_hash(a, b):
    """Order-independent hash for a pair of strings."""
    if a > b:
        a, b = b, a
    return hash((a, b))


def create_confidence_ladder(candidates, links_2025, random_baseline_precision, output_path):
    """
    Bar chart showing precision at different candidate tiers.
    Visual proof that confidence scoring is meaningful.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tier_defs = [
        ('Top 10', 10),
        ('Top 50', 50),
        ('Top 100', 100),
        ('Top 500', 500),
        ('Top 1K', 1000),
        ('Top 5K', 5000),
        ('Top 10K', 10000),
    ]

    labels = []
    precisions = []
    colors = []

    for label, k in tier_defs:
        if k > len(candidates):
            continue
        subset = candidates[:k]
        confirmed = sum(1 for c in subset
                        if _pair_hash(c['entity_a'].lower(), c['entity_b'].lower()) in links_2025)
        p = confirmed / len(subset)
        labels.append(label)
        precisions.append(p)
        colors.append('#1a73e8' if p > random_baseline_precision * 2 else '#90CAF9')

    # Add random baseline bar
    labels.append('Random\n(baseline)')
    precisions.append(random_baseline_precision)
    colors.append('#E0E0E0')

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(labels)), precisions, color=colors, edgecolor='white', linewidth=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Precision (fraction confirmed by 2025)', fontsize=12)
    ax.set_title('Higher Confidence → Higher Precision', fontsize=16, fontweight='bold')

    for bar, p in zip(bars, precisions):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f'{p:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.axhline(y=random_baseline_precision, color='red', linestyle='--',
               alpha=0.5, label=f'Random baseline: {random_baseline_precision:.2%}')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved confidence ladder: {output_path}")


# ═══════════════════════════════════════════════════════════
#  VISUALIZATION 2: Knowledge Frontier (Cross-Domain Map)
# ═══════════════════════════════════════════════════════════

def create_knowledge_frontier(future_predictions, entity_clusters, cluster_names,
                               output_path, top_n=200):
    """
    Network graph showing predicted cross-domain connections.
    The 'future of knowledge' visualization.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        import networkx as nx
    except ImportError:
        print("networkx not installed, skipping knowledge frontier visualization")
        return

    domain_pairs = Counter()
    example_pairs = defaultdict(list)

    for pred in future_predictions[:top_n]:
        ca = entity_clusters.get(pred['entity_a'], -1)
        cb = entity_clusters.get(pred['entity_b'], -1)
        if ca >= 0 and cb >= 0 and ca != cb:
            pair_key = tuple(sorted([ca, cb]))
            domain_pairs[pair_key] += 1
            if len(example_pairs[pair_key]) < 3:
                example_pairs[pair_key].append(
                    f"{pred['entity_a']} ↔ {pred['entity_b']}")

    G = nx.Graph()
    for (ca, cb), count in domain_pairs.most_common(25):
        name_a = ', '.join(cluster_names.get(ca, ['Unknown'])[:2])
        name_b = ', '.join(cluster_names.get(cb, ['Unknown'])[:2])
        G.add_edge(name_a, name_b, weight=count)

    if len(G.edges()) == 0:
        print("No cross-domain predictions found, skipping frontier visualization")
        return

    fig, ax = plt.subplots(figsize=(14, 14))
    pos = nx.spring_layout(G, k=2, seed=42)
    edges = G.edges(data=True)
    weights = [e[2]['weight'] for e in edges]
    max_w = max(weights) if weights else 1

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='#1a73e8', alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, width=[3 * w / max_w for w in weights],
                           alpha=0.4, edge_color='#FF6B6B', ax=ax)

    edge_labels = {(u, v): d['weight'] for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)

    ax.set_title("The Knowledge Frontier: Predicted Cross-Domain Connections (2025→Future)",
                 fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved knowledge frontier: {output_path}")

    return domain_pairs, example_pairs


# ═══════════════════════════════════════════════════════════
#  VISUALIZATION 3: Precision vs Threshold Curve
# ═══════════════════════════════════════════════════════════

def create_threshold_curve(threshold_results, random_baseline, output_path):
    """
    Line chart showing precision at different similarity thresholds.
    Shows where the 'signal' lives in the embedding space.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    thresholds = [r['threshold'] for r in threshold_results if r['n_candidates'] > 0]
    precisions = [r['precision'] for r in threshold_results if r['n_candidates'] > 0]
    counts = [r['n_candidates'] for r in threshold_results if r['n_candidates'] > 0]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = '#1a73e8'
    ax1.set_xlabel('Similarity Threshold', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12, color=color1)
    ax1.plot(thresholds, precisions, 'o-', color=color1, linewidth=2, markersize=8, label='Precision')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=random_baseline, color='red', linestyle='--', alpha=0.5,
                label=f'Random baseline: {random_baseline:.4f}')

    ax2 = ax1.twinx()
    color2 = '#888888'
    ax2.set_ylabel('Number of candidates', fontsize=12, color=color2)
    ax2.bar(thresholds, counts, width=0.03, alpha=0.2, color=color2, label='Candidate count')
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title('Precision vs Similarity Threshold', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved threshold curve: {output_path}")


# ═══════════════════════════════════════════════════════════
#  REAL-WORLD APPLICATION: Wikipedia Suggestions
# ═══════════════════════════════════════════════════════════

def generate_wikipedia_suggestions(future_predictions, validation_precision, top_n=100):
    """
    Format top predictions as Wikipedia editor suggestions.
    """
    suggestions = []
    for pred in future_predictions[:top_n]:
        a = pred['entity_a']
        b = pred['entity_b']
        suggestions.append({
            'source': a,
            'target': b,
            'confidence': pred.get('confidence', pred['similarity']),
            'similarity': pred['similarity'],
            'source_url': f"https://en.wikipedia.org/wiki/{a.replace(' ', '_')}",
            'target_url': f"https://en.wikipedia.org/wiki/{b.replace(' ', '_')}",
        })

    save_json(suggestions, f"{RESULTS_DIR}/wikipedia_suggestions.json")

    with open(f"{RESULTS_DIR}/wikipedia_suggestions.md", 'w') as f:
        f.write("# Suggested Wikipedia Link Additions\n\n")
        f.write(f"*Generated by Temporal Knowledge Gap Finder — {datetime.now():%Y-%m-%d}*\n")
        f.write(f"*Based on historical validation precision: ~{validation_precision:.1%}*\n\n")
        for i, s in enumerate(suggestions, 1):
            f.write(f"## {i}. [{s['source']}]({s['source_url']}) ↔ [{s['target']}]({s['target_url']})\n")
            f.write(f"Confidence: {s['confidence']:.3f} | Similarity: {s['similarity']:.3f}\n\n")

    print(f"Generated {len(suggestions)} Wikipedia suggestions")
    return suggestions


# ═══════════════════════════════════════════════════════════
#  RESEARCH TREND DETECTOR
# ═══════════════════════════════════════════════════════════

def research_trend_report(future_predictions, entity_clusters, cluster_names):
    """
    Identify the top emerging interdisciplinary research areas.
    """
    cross_domain = [p for p in future_predictions
                    if entity_clusters.get(p['entity_a'], -1) != entity_clusters.get(p['entity_b'], -1)
                    and entity_clusters.get(p['entity_a'], -1) >= 0]

    domain_connections = Counter()
    example_pairs = defaultdict(list)

    for p in cross_domain:
        ca = entity_clusters[p['entity_a']]
        cb = entity_clusters[p['entity_b']]
        pair_key = tuple(sorted([
            ', '.join(cluster_names.get(ca, ['?'])[:2]),
            ', '.join(cluster_names.get(cb, ['?'])[:2])
        ]))
        domain_connections[pair_key] += 1
        if len(example_pairs[pair_key]) < 3:
            example_pairs[pair_key].append(p)

    report = {
        'total_cross_domain': len(cross_domain),
        'total_predictions': len(future_predictions),
        'cross_domain_ratio': len(cross_domain) / max(len(future_predictions), 1),
        'top_emerging_areas': [],
    }

    print("\n" + "=" * 70)
    print("  EMERGING INTERDISCIPLINARY RESEARCH AREAS")
    print("=" * 70)

    for (d1, d2), count in domain_connections.most_common(15):
        area = {
            'domain_1': d1,
            'domain_2': d2,
            'prediction_count': count,
            'examples': [{'a': ex['entity_a'], 'b': ex['entity_b'],
                          'similarity': ex['similarity']}
                         for ex in example_pairs[(d1, d2)]],
        }
        report['top_emerging_areas'].append(area)

        print(f"\n  {d1}  ↔  {d2}  ({count} predicted connections)")
        for ex in example_pairs[(d1, d2)]:
            print(f"    - {ex['entity_a']} ↔ {ex['entity_b']} "
                  f"(sim: {ex['similarity']:.3f})")

    save_json(report, f"{RESULTS_DIR}/research_trends.json")
    return report


# ═══════════════════════════════════════════════════════════
#  BLOG-WORTHY EXAMPLES
# ═══════════════════════════════════════════════════════════

def generate_blog_examples(confirmed_pairs, n=10):
    """
    Select the most narratively compelling confirmed predictions.
    """
    scored = []
    for pair in confirmed_pairs:
        # Prefer high similarity, recognizable concepts
        narrative_score = pair['similarity']
        if pair.get('cross_domain', False):
            narrative_score += 0.1
        scored.append((narrative_score, pair))

    scored.sort(reverse=True)

    examples = []
    print("\nBLOG-WORTHY CONFIRMED PREDICTIONS:")
    print("(Verify each one manually — check Wikipedia edit history)\n")
    for score, pair in scored[:n]:
        a_url = pair['entity_a'].replace(' ', '_')
        b_url = pair['entity_b'].replace(' ', '_')
        print(f"  '{pair['entity_a']}' ↔ '{pair['entity_b']}'")
        print(f"  Similarity: {pair['similarity']:.3f}")
        print(f"  A: https://en.wikipedia.org/wiki/{a_url}")
        print(f"  B: https://en.wikipedia.org/wiki/{b_url}")
        print(f"  History: https://en.wikipedia.org/w/index.php?title={a_url}&action=history")
        print()
        examples.append(pair)

    save_json(examples, f"{RESULTS_DIR}/blog_examples.json")
    return examples


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  STEP 6: VISUALIZATION & APPLICATIONS")
    print("=" * 60)
    log_memory("start")

    # Load results from previous steps
    validation_path = f"{RESULTS_DIR}/validation_results.json"
    candidates_path = f"{RESULTS_DIR}/candidates_2015.json"
    future_path = f"{RESULTS_DIR}/future_predictions_2025.json"

    if not os.path.exists(validation_path):
        print("No validation results found. Run step 5 first.")
        return

    validation = load_json(validation_path)  # list of candidate dicts
    candidates = load_json(candidates_path)

    # Load significance results for random baseline
    sig_path = f"{RESULTS_DIR}/significance_results.json"
    sig_results = load_json(sig_path) if os.path.exists(sig_path) else {}

    # ── Confidence Ladder ──
    print("\n--- Creating confidence ladder ---")
    links_2025 = set()
    links_path = f"{DATA_DIR}/links_2025.json"
    if os.path.exists(links_path):
        link_dict = load_json(links_path)
        for source, targets in link_dict.items():
            s = source.lower()
            for target in targets:
                links_2025.add(_pair_hash(s, target.lower()))

    random_baseline = sig_results.get('random_mean', 0.001)
    create_confidence_ladder(candidates, links_2025, random_baseline,
                             f"{RESULTS_DIR}/confidence_ladder.png")

    del links_2025
    gc.collect()

    # ── Threshold Curve ──
    print("\n--- Creating threshold curve ---")
    threshold_path = f"{RESULTS_DIR}/threshold_analysis.json"
    threshold_results = load_json(threshold_path) if os.path.exists(threshold_path) else []
    if threshold_results:
        create_threshold_curve(threshold_results, random_baseline,
                                f"{RESULTS_DIR}/threshold_curve.png")

    # ── Blog Examples ──
    print("\n--- Selecting blog-worthy examples ---")
    confirmed = [c for c in validation if c.get('linked_2025') or c.get('prediction_correct')]
    if confirmed:
        generate_blog_examples(confirmed)

    # ── Future Predictions Visualizations ──
    if os.path.exists(future_path):
        future = load_json(future_path)

        clusters_path = f"{RESULTS_DIR}/entity_clusters.json"
        cluster_names_path = f"{RESULTS_DIR}/cluster_names.json"

        if os.path.exists(clusters_path) and os.path.exists(cluster_names_path):
            entity_clusters = load_json(clusters_path)
            cluster_names = {int(k): v for k, v in load_json(cluster_names_path).items()}

            print("\n--- Creating knowledge frontier ---")
            create_knowledge_frontier(future, entity_clusters, cluster_names,
                                       f"{RESULTS_DIR}/knowledge_frontier.png")

            print("\n--- Generating research trend report ---")
            research_trend_report(future, entity_clusters, cluster_names)

        print("\n--- Generating Wikipedia suggestions ---")
        val_precision = sig_results.get('embedding_precision', 0.025)
        generate_wikipedia_suggestions(future, val_precision)

    log_memory("end")
    print("\n✅ Step 6 complete: Visualizations and applications generated")


if __name__ == '__main__':
    main()
