"""
validate_predictions.py — Phase 4: Validate hidden-connection candidates against
a later Wikipedia link graph to measure predictive power.

Usage:
    python scripts/validate_predictions.py \\
        --candidates  results/candidates_2018.json \\
        --links-later data/links_2025.json \\
        [--links-base data/links_2018.json]   # for random baseline

Outputs:
    - Precision@K table printed to stdout
    - results/precision_at_k.png  (if matplotlib available)
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from parse_pagelinks import load_link_set


# ── Core validation ───────────────────────────────────────────────────────────

def is_now_linked(candidate, link_set_later):
    a = candidate["entity_a"].lower()
    b = candidate["entity_b"].lower()
    return (a, b) in link_set_later or (b, a) in link_set_later


def validate_predictions(candidates, link_set_later):
    """
    Check how many candidates became explicit Wikipedia links in the later dump.

    Returns dict with 'total', 'confirmed', 'precision', and per-candidate lists.
    """
    total = len(candidates)
    confirmed_pairs = []
    unconfirmed_pairs = []

    for c in candidates:
        if is_now_linked(c, link_set_later):
            confirmed_pairs.append({**c, "confirmed": True})
        else:
            unconfirmed_pairs.append({**c, "confirmed": False})

    confirmed = len(confirmed_pairs)
    return {
        "total_predictions": total,
        "confirmed": confirmed,
        "precision": confirmed / total if total > 0 else 0.0,
        "confirmed_pairs": confirmed_pairs,
        "unconfirmed_pairs": unconfirmed_pairs,
    }


def precision_at_k(candidates, link_set_later, k_values):
    """
    Calculate precision at each K value.

    Candidates should already be sorted by similarity descending.

    Returns list of {'k': int, 'precision': float, 'confirmed': int}
    """
    results = []
    for k in k_values:
        top_k = candidates[:k]
        confirmed = sum(1 for c in top_k if is_now_linked(c, link_set_later))
        precision = confirmed / k if k > 0 else 0.0
        results.append({"k": k, "precision": precision, "confirmed": confirmed})
        print(f"  Precision@{k:>7,}: {precision:.4f}  ({confirmed}/{k} confirmed)")
    return results


def random_baseline_precision(link_set_base, link_set_later,
                               n_samples=10000, n_trials=5):
    """
    Estimate precision for randomly selected unlinked entity pairs.
    This is the 'chance' level to beat.

    link_set_base: pairs that existed at the earlier time (exclude these — must be unlinked)
    link_set_later: pairs that exist at the later time (these count as 'confirmed')
    """
    # Collect all known entity titles from the base link set
    entity_set = set()
    for a, b in link_set_base:
        entity_set.add(a)
        entity_set.add(b)
    entity_list = list(entity_set)

    if len(entity_list) < 2:
        print("Not enough entities in base link set for random baseline.")
        return None

    precisions = []
    print(f"\nComputing random baseline ({n_trials} trials × {n_samples:,} samples) ...")
    for trial in range(n_trials):
        confirmed = 0
        attempts = 0
        while confirmed + (n_samples - attempts) > 0 and attempts < n_samples * 10:
            a = random.choice(entity_list)
            b = random.choice(entity_list)
            if a == b:
                continue
            # Must be unlinked in the base year
            if (a, b) in link_set_base or (b, a) in link_set_base:
                continue
            attempts += 1
            if (a, b) in link_set_later or (b, a) in link_set_later:
                confirmed += 1
            if attempts >= n_samples:
                break
        precision = confirmed / attempts if attempts > 0 else 0.0
        precisions.append(precision)

    avg = sum(precisions) / len(precisions)
    spread = max(precisions) - min(precisions)
    print(f"  Random baseline precision: {avg:.5f} (±{spread:.5f})")
    return avg


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_precision_at_k(pak_results, baseline=None, output_path="results/precision_at_k.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    ks = [r["k"] for r in pak_results]
    precisions = [r["precision"] for r in pak_results]

    plt.figure(figsize=(10, 6))
    plt.plot(ks, precisions, "bo-", linewidth=2, markersize=8, label="Embedding predictions")

    if baseline is not None:
        plt.axhline(baseline, color="red", linestyle="--", linewidth=1.5,
                    label=f"Random baseline ({baseline:.4f})")

    plt.xlabel("K (number of top predictions)", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision@K: Do Higher-Confidence Predictions Come True More Often?", fontsize=13)
    plt.xscale("log")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--candidates", required=True,
                        help="JSON file of candidates from find_hidden_connections.py")
    parser.add_argument("--links-later", required=True,
                        help="Link JSON for the later time point (e.g. 2025)")
    parser.add_argument("--links-base", default=None,
                        help="Link JSON for the base time point (e.g. 2018) — used for random baseline")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot")
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates} ...")
    with open(args.candidates) as f:
        candidates = json.load(f)
    # Ensure sorted by similarity descending
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    print(f"  {len(candidates):,} candidates loaded")

    print(f"Loading later link set from {args.links_later} ...")
    link_set_later = load_link_set(args.links_later)
    print(f"  {len(link_set_later):,} directed pairs")

    # ── Overall precision ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  VALIDATION RESULTS")
    print("=" * 55)
    result = validate_predictions(candidates, link_set_later)
    print(f"  Total predictions:  {result['total_predictions']:>10,}")
    print(f"  Confirmed by later: {result['confirmed']:>10,}")
    print(f"  Overall precision:  {result['precision']:>10.2%}")
    print("=" * 55)

    # ── Precision@K ────────────────────────────────────────────────────────────
    k_values = [k for k in [10, 50, 100, 500, 1000, 5000, 10000] if k <= len(candidates)]
    if k_values:
        print(f"\nPrecision@K (candidates sorted by similarity):")
        pak_results = precision_at_k(candidates, link_set_later, k_values)
    else:
        pak_results = []

    # ── Top confirmed pairs ────────────────────────────────────────────────────
    confirmed = result["confirmed_pairs"]
    print(f"\nTop confirmed connections (predicted earlier, linked later):")
    for pair in confirmed[:20]:
        print(f"  ✓ {pair['entity_a']!r:40s} <-> {pair['entity_b']!r}  sim={pair['similarity']:.3f}")

    # ── Random baseline ────────────────────────────────────────────────────────
    baseline = None
    if args.links_base:
        print(f"\nLoading base link set from {args.links_base} ...")
        link_set_base = load_link_set(args.links_base)
        print(f"  {len(link_set_base):,} directed pairs")
        baseline = random_baseline_precision(link_set_base, link_set_later)
        if baseline is not None and pak_results:
            p100 = next((r["precision"] for r in pak_results if r["k"] == 100), None)
            if p100 is not None and baseline > 0:
                print(f"\n  Precision@100 vs baseline: {p100:.4f} vs {baseline:.5f} "
                      f"= {p100/baseline:.1f}x improvement")

    # ── Plot ───────────────────────────────────────────────────────────────────
    if pak_results and not args.no_plot:
        plot_precision_at_k(pak_results, baseline=baseline)

    # ── Save full results ──────────────────────────────────────────────────────
    output_json = args.candidates.replace("candidates_", "validation_")
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull results saved to {output_json}")


if __name__ == "__main__":
    main()
