"""Validate 2015 predictions against 2025 reality.
Improvements applied:
  - Stratified validation: existing_to_existing vs involves_new vs involves_missing (Iteration 2)
  - Multi-threshold precision analysis (Iteration 1, fix #3)
  - Statistical significance with z-test and binomial test (Iteration 1, fix #5)
  - Shortest path analysis for top candidates via BFS (Iteration 1, fix #4)
  - Auto-generated HTML report (Iteration 3)
  - Memory-safe phased execution (Iteration 2)
"""

import gc
import json
import math
import os
import random
import sys
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    EXPERIMENT_CONFIG, DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR,
    init_dirs, set_all_seeds, log_memory, save_run_manifest,
)

ANALYSIS = EXPERIMENT_CONFIG['analysis']


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_json(path):
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    """Save data as JSON."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def _pair_hash(a, b):
    """Order-independent hash for a pair of strings."""
    if a > b:
        a, b = b, a
    return hash((a, b))


def load_link_set(links_path):
    """Load link graph JSON as a set of integer hashes (memory-efficient).

    Uses order-independent hashing so (A,B) and (B,A) map to the same int.
    """
    link_dict = load_json(links_path)
    link_set = set()
    for source, targets in link_dict.items():
        s = source.lower()
        for target in targets:
            link_set.add(_pair_hash(s, target.lower()))
    return link_set


def load_link_graph(links_path):
    """Load link graph JSON as a dict (lowercase keys -> list of lowercase targets)."""
    raw = load_json(links_path)
    graph = {}
    for source, targets in raw.items():
        graph[source.lower()] = [t.lower() for t in targets]
    return graph


def build_entity_set(links_path):
    """Build the set of all entity titles (lowercase) present in a link graph."""
    link_dict = load_json(links_path)
    entities = set()
    for source, targets in link_dict.items():
        entities.add(source.lower())
        for t in targets:
            entities.add(t.lower())
    return entities


def is_now_linked(candidate, link_set_2025):
    """Check whether a candidate pair is linked in the 2025 graph."""
    a = candidate['entity_a'].lower()
    b = candidate['entity_b'].lower()
    return _pair_hash(a, b) in link_set_2025


# ---------------------------------------------------------------------------
# Shortest path via BFS (Iteration 1, fix #4)
# ---------------------------------------------------------------------------

def shortest_path_length(source, target, link_graph, max_depth=None):
    """BFS shortest path length. Returns path length or -1 if not found."""
    if max_depth is None:
        max_depth = ANALYSIS['shortest_path_max_depth']
    source = source.lower()
    target = target.lower()
    if source == target:
        return 0
    visited = {source}
    queue = deque([(source, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in link_graph.get(node, []):
            if neighbor == target:
                return depth + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return -1


def compute_shortest_paths(candidates, link_graph, top_n=None):
    """Compute shortest path lengths for the top-N candidates."""
    if top_n is None:
        top_n = ANALYSIS['shortest_path_top_n']
    subset = candidates[:top_n]
    print(f"Computing shortest paths for top {len(subset)} candidates "
          f"(max_depth={ANALYSIS['shortest_path_max_depth']})...")

    path_lengths = []
    for c in tqdm(subset, desc="BFS shortest paths"):
        pl = shortest_path_length(c['entity_a'], c['entity_b'], link_graph)
        c['shortest_path_2015'] = pl
        path_lengths.append(pl)

    dist = defaultdict(int)
    for pl in path_lengths:
        label = str(pl) if pl >= 0 else "disconnected"
        dist[label] += 1

    print("Shortest path distribution (top candidates):")
    for k in sorted(dist.keys()):
        print(f"  path_length={k}: {dist[k]}")

    return subset, dict(dist)


# ---------------------------------------------------------------------------
# Stratified validation (Iteration 2)
# ---------------------------------------------------------------------------

def stratified_validation(candidates, link_set_2025,
                          entities_2015_set, entities_2025_set):
    """Validate predictions with stratification by WHY a link appeared.

    Categories:
      existing_to_existing — both articles existed in 2015 and still exist in 2025
      involves_new_article — at least one article is new (present in 2025 but not 2015)
      involves_missing     — at least one article is gone (present in 2015 but not 2025)
    """
    results = {
        'existing_to_existing': {'total': 0, 'confirmed': 0, 'pairs': []},
        'involves_new_article': {'total': 0, 'confirmed': 0, 'pairs': []},
        'involves_missing':     {'total': 0, 'confirmed': 0, 'pairs': []},
    }

    validated = []
    for c in candidates:
        a = c['entity_a'].lower()
        b = c['entity_b'].lower()
        a_in_2015 = a in entities_2015_set
        b_in_2015 = b in entities_2015_set
        a_in_2025 = a in entities_2025_set
        b_in_2025 = b in entities_2025_set

        if not a_in_2025 or not b_in_2025:
            category = 'involves_missing'
        elif not a_in_2015 or not b_in_2015:
            category = 'involves_new_article'
        else:
            category = 'existing_to_existing'

        now_linked = is_now_linked(c, link_set_2025)
        results[category]['total'] += 1
        if now_linked:
            results[category]['confirmed'] += 1
            results[category]['pairs'].append(c)

        validated.append({
            **c,
            'linked_2025': now_linked,
            'prediction_correct': now_linked,
            'category': category,
        })

    for cat, data in results.items():
        data['precision'] = (data['confirmed'] / data['total']
                             if data['total'] > 0 else 0.0)

    return validated, results


# ---------------------------------------------------------------------------
# Multi-threshold precision analysis (Iteration 1, fix #3)
# ---------------------------------------------------------------------------

def multi_threshold_analysis(validated, thresholds=None):
    """Compute precision at multiple similarity thresholds."""
    if thresholds is None:
        thresholds = ANALYSIS['similarity_thresholds']

    results = []
    for thresh in thresholds:
        above = [v for v in validated if v['similarity'] >= thresh]
        if not above:
            results.append({
                'threshold': thresh,
                'n_candidates': 0,
                'confirmed': 0,
                'precision': 0.0,
            })
            continue
        confirmed = sum(1 for v in above if v['prediction_correct'])
        results.append({
            'threshold': thresh,
            'n_candidates': len(above),
            'confirmed': confirmed,
            'precision': confirmed / len(above),
        })

    print("\nMulti-threshold precision analysis:")
    print(f"  {'Threshold':<12} {'Candidates':<12} {'Confirmed':<12} {'Precision':<12}")
    print("  " + "-" * 48)
    for r in results:
        print(f"  {r['threshold']:<12.2f} {r['n_candidates']:<12} "
              f"{r['confirmed']:<12} {r['precision']:<12.4f}")

    return results


# ---------------------------------------------------------------------------
# Statistical significance testing (Iteration 1, fix #5)
# ---------------------------------------------------------------------------

def random_baseline(all_entity_titles, link_set_2025,
                    n_samples=None, n_trials=None):
    """Compute random baseline precision by sampling random entity pairs."""
    if n_samples is None:
        n_samples = ANALYSIS['random_baseline_samples']
    if n_trials is None:
        n_trials = ANALYSIS['random_baseline_trials']

    titles = list(all_entity_titles)
    precisions = []

    print(f"Running {n_trials} random baseline trials ({n_samples} pairs each)...")
    for trial in range(n_trials):
        confirmed = 0
        for _ in range(n_samples):
            a = random.choice(titles)
            b = random.choice(titles)
            if a == b:
                continue
            if _pair_hash(a, b) in link_set_2025:
                confirmed += 1
        precisions.append(confirmed / n_samples)

    mean_prec = float(np.mean(precisions))
    std_prec = float(np.std(precisions))
    print(f"Random baseline: mean={mean_prec:.6f}, std={std_prec:.6f}")
    return precisions, mean_prec, std_prec


def statistical_significance(embedding_precision, random_precisions,
                             n_predictions):
    """Test whether embedding-based precision significantly exceeds random.

    Returns z-test, binomial test, and Cohen's h effect size.
    """
    random_mean = float(np.mean(random_precisions))
    random_std = float(np.std(random_precisions))

    # Z-test
    if random_std > 0 and len(random_precisions) > 1:
        z_score = ((embedding_precision - random_mean)
                   / (random_std / math.sqrt(len(random_precisions))))
        p_value_z = float(1 - stats.norm.cdf(z_score))
    else:
        z_score = float('inf') if embedding_precision > random_mean else 0.0
        p_value_z = 0.0 if embedding_precision > random_mean else 1.0

    # Binomial test
    n_confirmed = int(round(embedding_precision * n_predictions))
    p_baseline = max(random_mean, 1e-10)
    binom_result = stats.binomtest(n_confirmed, n_predictions, p_baseline,
                                   alternative='greater')
    p_value_binom = float(binom_result.pvalue)

    # Cohen's h for proportions
    h = (2 * math.asin(math.sqrt(max(0, min(1, embedding_precision))))
         - 2 * math.asin(math.sqrt(max(0, min(1, random_mean)))))

    if abs(h) > 0.8:
        effect_label = 'large'
    elif abs(h) > 0.5:
        effect_label = 'medium'
    else:
        effect_label = 'small'

    result = {
        'embedding_precision': embedding_precision,
        'random_mean': random_mean,
        'random_std': random_std,
        'z_score': float(z_score),
        'p_value_z': p_value_z,
        'p_value_binom': p_value_binom,
        'effect_size_h': float(h),
        'effect_interpretation': effect_label,
        'significant_at_001': p_value_z < 0.001,
        'significant_at_005': p_value_z < 0.05,
        'n_predictions': n_predictions,
        'n_confirmed': n_confirmed,
    }

    print("\nStatistical significance:")
    print(f"  Embedding precision: {embedding_precision:.6f}")
    print(f"  Random baseline:     {random_mean:.6f} +/- {random_std:.6f}")
    print(f"  Z-score:             {z_score:.2f}")
    print(f"  P-value (z-test):    {p_value_z:.2e}")
    print(f"  P-value (binomial):  {p_value_binom:.2e}")
    print(f"  Effect size (h):     {h:.3f} ({effect_label})")
    print(f"  Significant at 0.1%: {'YES' if result['significant_at_001'] else 'NO'}")

    return result


# ---------------------------------------------------------------------------
# Similarity bucket breakdown
# ---------------------------------------------------------------------------

def similarity_bucket_breakdown(validated):
    """Break results into similarity buckets."""
    buckets = defaultdict(lambda: {'total': 0, 'correct': 0})
    for v in validated:
        sim = v['similarity']
        if sim >= 0.80:
            bucket = '0.80+'
        elif sim >= 0.75:
            bucket = '0.75-0.80'
        elif sim >= 0.70:
            bucket = '0.70-0.75'
        elif sim >= 0.65:
            bucket = '0.65-0.70'
        elif sim >= 0.60:
            bucket = '0.60-0.65'
        elif sim >= 0.55:
            bucket = '0.55-0.60'
        else:
            bucket = '<0.55'
        buckets[bucket]['total'] += 1
        if v['prediction_correct']:
            buckets[bucket]['correct'] += 1

    print(f"\n{'Similarity Range':<20} {'Total':<10} {'Correct':<10} {'Precision':<10}")
    print('-' * 50)
    for bkt in sorted(buckets.keys(), reverse=True):
        b = buckets[bkt]
        p = b['correct'] / b['total'] if b['total'] > 0 else 0
        print(f"{bkt:<20} {b['total']:<10} {b['correct']:<10} {p:.4f}")

    return dict(buckets)


# ---------------------------------------------------------------------------
# HTML report generation (Iteration 3)
# ---------------------------------------------------------------------------

def generate_html_report(validated, strat_results, threshold_results,
                         significance, path_distribution, run_manifest,
                         output_path):
    """Generate a self-contained HTML report with all findings."""

    # Build validation breakdown table rows
    strat_rows = ""
    for cat in ['existing_to_existing', 'involves_new_article', 'involves_missing']:
        d = strat_results[cat]
        prec_str = f"{d['precision']:.4f}" if d['total'] > 0 else "N/A"
        strat_rows += (f"<tr><td>{cat}</td><td>{d['total']}</td>"
                       f"<td>{d['confirmed']}</td><td>{prec_str}</td></tr>\n")

    # Multi-threshold table rows
    thresh_rows = ""
    for r in threshold_results:
        prec_str = f"{r['precision']:.4f}" if r['n_candidates'] > 0 else "N/A"
        thresh_rows += (f"<tr><td>{r['threshold']:.2f}</td>"
                        f"<td>{r['n_candidates']}</td>"
                        f"<td>{r['confirmed']}</td>"
                        f"<td>{prec_str}</td></tr>\n")

    # Top confirmed predictions
    correct_preds = [v for v in validated if v['prediction_correct']]
    correct_preds.sort(key=lambda x: x['similarity'], reverse=True)
    top_confirmed_rows = ""
    for c in correct_preds[:30]:
        cat_label = c.get('category', '')
        sp = c.get('shortest_path_2015', '')
        top_confirmed_rows += (
            f"<tr><td>{c['entity_a']}</td><td>{c['entity_b']}</td>"
            f"<td>{c['similarity']:.3f}</td>"
            f"<td>{cat_label}</td><td>{sp}</td></tr>\n"
        )

    # Path distribution
    path_rows = ""
    if path_distribution:
        for k in sorted(path_distribution.keys()):
            path_rows += f"<tr><td>{k}</td><td>{path_distribution[k]}</td></tr>\n"

    # Significance summary
    sig = significance
    sig_class = "confirmed" if sig.get('significant_at_001') else "prediction"
    sig_label = "YES" if sig.get('significant_at_001') else "NO"

    # Pure precision
    pure_prec = strat_results['existing_to_existing']['precision']
    pure_total = strat_results['existing_to_existing']['total']
    pure_confirmed = strat_results['existing_to_existing']['confirmed']

    overall_total = len(validated)
    overall_confirmed = sum(1 for v in validated if v['prediction_correct'])
    overall_prec = overall_confirmed / overall_total if overall_total > 0 else 0

    # Config block
    config_json = json.dumps(run_manifest.get('config', EXPERIMENT_CONFIG), indent=2)

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Temporal Knowledge Gap Finder - Results</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 960px; margin: 40px auto;
         color: #2d2d2d; line-height: 1.7; padding: 0 20px; }}
  h1 {{ border-bottom: 3px solid #333; padding-bottom: 10px; }}
  h2 {{ margin-top: 40px; color: #1a1a1a; }}
  .metric {{ font-size: 2em; font-weight: bold; color: #1a73e8; }}
  .card {{ background: #f8f9fa; border-radius: 8px; padding: 20px;
           margin: 16px 0; border-left: 4px solid #1a73e8; }}
  .card-red {{ border-left-color: #c5221f; }}
  .card-green {{ border-left-color: #0d904f; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  .confirmed {{ color: #0d904f; font-weight: bold; }}
  .prediction {{ color: #c5221f; }}
  pre {{ background: #f4f4f4; padding: 16px; border-radius: 6px;
         overflow-x: auto; font-size: 0.85em; }}
  .two-col {{ display: flex; gap: 20px; }}
  .two-col > .card {{ flex: 1; }}
</style>
</head><body>

<h1>Temporal Knowledge Gap Finder</h1>
<p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</em></p>

<h2>Key Results</h2>
<div class="two-col">
  <div class="card card-green">
    <div class="metric">{pure_prec:.1%}</div>
    <p><strong>Pure precision</strong> &mdash; both articles existed in 2015, new link appeared by 2025.<br>
    {pure_confirmed} confirmed out of {pure_total} predictions.</p>
  </div>
  <div class="card">
    <div class="metric">{overall_prec:.1%}</div>
    <p><strong>Overall precision</strong> &mdash; all categories combined.<br>
    {overall_confirmed} confirmed out of {overall_total} predictions.</p>
  </div>
</div>

<h2>Statistical Significance</h2>
<div class="card {'card-green' if sig.get('significant_at_001') else 'card-red'}">
  <p>Significant at 0.1% level: <span class="{sig_class}">{sig_label}</span></p>
  <table>
    <tr><td>Embedding precision</td><td>{sig.get('embedding_precision', 0):.6f}</td></tr>
    <tr><td>Random baseline</td><td>{sig.get('random_mean', 0):.6f} &plusmn; {sig.get('random_std', 0):.6f}</td></tr>
    <tr><td>Z-score</td><td>{sig.get('z_score', 0):.2f}</td></tr>
    <tr><td>P-value (z-test)</td><td>{sig.get('p_value_z', 1):.2e}</td></tr>
    <tr><td>P-value (binomial)</td><td>{sig.get('p_value_binom', 1):.2e}</td></tr>
    <tr><td>Effect size (Cohen's h)</td><td>{sig.get('effect_size_h', 0):.3f} ({sig.get('effect_interpretation', 'N/A')})</td></tr>
  </table>
</div>

<h2>Validation Breakdown by Category</h2>
<table>
  <tr><th>Category</th><th>Predictions</th><th>Confirmed</th><th>Precision</th></tr>
  {strat_rows}
</table>

<h2>Precision at Multiple Similarity Thresholds</h2>
<table>
  <tr><th>Threshold</th><th>Candidates</th><th>Confirmed</th><th>Precision</th></tr>
  {thresh_rows}
</table>

<h2>Shortest Path Distribution (Top Candidates)</h2>
<table>
  <tr><th>Path Length</th><th>Count</th></tr>
  {path_rows}
</table>

<h2>Top 30 Confirmed Predictions (2015 to 2025)</h2>
<table>
  <tr><th>Entity A</th><th>Entity B</th><th>Similarity</th><th>Category</th><th>Path (2015)</th></tr>
  {top_confirmed_rows}
</table>

<h2>Methodology &amp; Reproducibility</h2>
<pre>{config_json}</pre>

</body></html>"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"HTML report generated: {output_path}")


# ---------------------------------------------------------------------------
# Main — memory-safe phased execution
# ---------------------------------------------------------------------------

def run_validation():
    """Run the complete validation pipeline in memory-safe phases."""
    init_dirs()
    set_all_seeds()

    # Save manifest at start
    manifest = save_run_manifest()
    log_memory("Start of validation")

    # ── PHASE 1: Load candidates ──
    print("\n=== Phase 1: Loading candidate predictions ===")
    candidates_path = f"{RESULTS_DIR}/candidates_2015.json"
    candidates = load_json(candidates_path)
    print(f"Loaded {len(candidates)} candidates from 2015 analysis")

    # ── PHASE 2: Load link sets and entity sets ──
    print("\n=== Phase 2: Loading link sets and entity sets ===")
    log_memory("Before loading link data")

    link_set_2025 = load_link_set(f"{DATA_DIR}/links_2025.json")
    entities_2015_set = build_entity_set(f"{DATA_DIR}/links_2015.json")
    entities_2025_set = build_entity_set(f"{DATA_DIR}/links_2025.json")
    log_memory("After loading link sets")

    # ── PHASE 3: Stratified validation ──
    print("\n=== Phase 3: Stratified validation ===")
    validated, strat_results = stratified_validation(
        candidates, link_set_2025, entities_2015_set, entities_2025_set
    )

    total = len(validated)
    correct = sum(1 for v in validated if v['prediction_correct'])
    overall_precision = correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Total predictions:          {total}")
    print(f"Became linked by 2025:      {correct}")
    print(f"Overall precision:          {overall_precision:.4f} ({overall_precision*100:.2f}%)")
    print()
    for cat in ['existing_to_existing', 'involves_new_article', 'involves_missing']:
        d = strat_results[cat]
        p = f"{d['precision']:.4f}" if d['total'] > 0 else "N/A"
        print(f"  {cat:<30} {d['total']:>8} total, "
              f"{d['confirmed']:>8} confirmed, precision={p}")

    # Save intermediate
    save_json(validated, f"{RESULTS_DIR}/validation_results.json")
    save_json(strat_results, f"{RESULTS_DIR}/stratified_results.json")

    # ── PHASE 4: Multi-threshold analysis ──
    print("\n=== Phase 4: Multi-threshold precision analysis ===")
    threshold_results = multi_threshold_analysis(validated)
    save_json(threshold_results, f"{RESULTS_DIR}/threshold_analysis.json")

    # Similarity bucket breakdown
    bucket_data = similarity_bucket_breakdown(validated)
    save_json(bucket_data, f"{RESULTS_DIR}/similarity_buckets.json")

    # ── PHASE 5: Statistical significance ──
    print("\n=== Phase 5: Statistical significance testing ===")

    # Build the set of all entity titles for random sampling
    all_entity_titles = list(entities_2015_set & entities_2025_set)
    random_precs, rand_mean, rand_std = random_baseline(
        all_entity_titles, link_set_2025
    )

    # Use the "pure" existing_to_existing precision for the main test
    pure_prec = strat_results['existing_to_existing']['precision']
    pure_total = strat_results['existing_to_existing']['total']
    sig_result = statistical_significance(pure_prec, random_precs, pure_total)
    save_json(sig_result, f"{RESULTS_DIR}/significance_results.json")

    # Free link set to make room for link graph
    del link_set_2025, entities_2015_set, entities_2025_set, all_entity_titles
    gc.collect()
    log_memory("After freeing link sets")

    # ── PHASE 6: Shortest path analysis ──
    print("\n=== Phase 6: Shortest path analysis (BFS on 2015 link graph) ===")
    log_memory("Before loading link graph for BFS")
    link_graph_2015 = load_link_graph(f"{DATA_DIR}/links_2015.json")
    log_memory("After loading link graph")

    top_with_paths, path_distribution = compute_shortest_paths(
        validated, link_graph_2015
    )
    save_json(top_with_paths, f"{RESULTS_DIR}/top_candidates_with_paths.json")
    save_json(path_distribution, f"{RESULTS_DIR}/path_distribution.json")

    # Show top correct predictions
    correct_preds = [v for v in validated if v['prediction_correct']]
    correct_preds.sort(key=lambda x: x['similarity'], reverse=True)
    print(f"\nTop 20 CORRECT predictions (hidden in 2015, linked by 2025):")
    for c in correct_preds[:20]:
        sp = c.get('shortest_path_2015', '?')
        print(f"  {c['entity_a']} <--> {c['entity_b']}  "
              f"(sim: {c['similarity']:.3f}, path: {sp}, cat: {c['category']})")

    del link_graph_2015
    gc.collect()
    log_memory("After freeing link graph")

    # ── PHASE 7: Generate HTML report ──
    print("\n=== Phase 7: Generating HTML report ===")
    generate_html_report(
        validated, strat_results, threshold_results,
        sig_result, path_distribution, manifest,
        f"{RESULTS_DIR}/report.html"
    )

    log_memory("Validation complete")
    print(f"\n{'='*60}")
    print("All results saved to:", RESULTS_DIR)
    print(f"{'='*60}")

    return validated, strat_results


if __name__ == "__main__":
    run_validation()
