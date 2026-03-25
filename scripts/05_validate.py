"""Validate 2015 predictions against 2025 reality."""

import json
import os
from collections import defaultdict

def validate_predictions(candidates_2015_path, links_2025_path, output_path):
    """Check which 2015 'hidden connections' became real links by 2025."""
    
    with open(candidates_2015_path) as f:
        candidates = json.load(f)
    
    with open(links_2025_path) as f:
        link_dict = json.load(f)
    
    # Build 2025 link set
    link_set_2025 = set()
    for source, targets in link_dict.items():
        for target in targets:
            link_set_2025.add((source.lower(), target.lower()))
            link_set_2025.add((target.lower(), source.lower()))
    
    # Check each candidate
    validated = []
    for c in candidates:
        a = c['entity_a'].lower()
        b = c['entity_b'].lower()
        now_linked = (a, b) in link_set_2025 or (b, a) in link_set_2025
        
        validated.append({
            **c,
            'linked_2025': now_linked,
            'prediction_correct': now_linked  # unlinked in 2015, linked in 2025
        })
    
    # Save full results
    with open(output_path, 'w') as f:
        json.dump(validated, f, indent=2)
    
    # ── Compute metrics ──
    total = len(validated)
    correct = sum(1 for v in validated if v['prediction_correct'])
    precision = correct / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Total predictions:          {total}")
    print(f"Became linked by 2025:      {correct}")
    print(f"Precision (hit rate):       {precision:.4f} ({precision*100:.2f}%)")
    
    # ── Breakdown by similarity buckets ──
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
        else:
            bucket = '0.60-0.65'
        
        buckets[bucket]['total'] += 1
        buckets[bucket]['correct'] += 1 if v['prediction_correct'] else 0
    
    print(f"\n{'Similarity Range':<20} {'Total':<10} {'Correct':<10} {'Precision':<10}")
    print('-' * 50)
    for bucket in sorted(buckets.keys(), reverse=True):
        b = buckets[bucket]
        p = b['correct'] / b['total'] if b['total'] > 0 else 0
        print(f"{bucket:<20} {b['total']:<10} {b['correct']:<10} {p:.4f}")
    
    # ── Show top correct predictions ──
    correct_preds = [v for v in validated if v['prediction_correct']]
    correct_preds.sort(key=lambda x: x['similarity'], reverse=True)
    
    print(f"\nTop 20 CORRECT predictions (hidden in 2015, linked by 2025):")
    for c in correct_preds[:20]:
        print(f"  {c['entity_a']} <──> {c['entity_b']}  (similarity: {c['similarity']:.3f})")
    
    return validated

if __name__ == "__main__":
    data_dir = os.path.expanduser("~/data")
    results_dir = os.path.expanduser("~/data/results")
    
    validate_predictions(
        f"{results_dir}/candidates_2015.json",
        f"{data_dir}/links_2025.json",
        f"{results_dir}/validation_results.json"
    )
