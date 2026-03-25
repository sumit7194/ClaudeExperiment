"""Find concept pairs close in embedding space but NOT linked.
Saves progress every 1000 entities — fully resumable."""

import json
import os
import pickle
import numpy as np
from wikipedia2vec import Wikipedia2Vec
from tqdm import tqdm

SAVE_INTERVAL = 1000  # Save every 1000 entities

def load_link_set(links_path):
    """Load link graph as a set of (source, target) pairs."""
    with open(links_path) as f:
        link_dict = json.load(f)
    link_set = set()
    for source, targets in link_dict.items():
        for target in targets:
            link_set.add((source.lower(), target.lower()))
            link_set.add((target.lower(), source.lower()))
    return link_set

def are_linked(entity_a, entity_b, link_set):
    a = entity_a.title.lower()
    b = entity_b.title.lower()
    return (a, b) in link_set or (b, a) in link_set

def find_close_but_unlinked(model, link_set, output_path, checkpoint_path,
                             similarity_threshold=0.60,
                             max_pairs=50000,
                             min_entity_name_length=3,
                             num_entities=20000):
    """Find hidden connections with periodic checkpointing."""
    
    entities = [e for e in model.dictionary.entities() 
                if len(e.title) >= min_entity_name_length]
    entities = entities[:num_entities]
    
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
            print(f"  Loaded {len(candidates)} candidates, resuming from entity {start_idx}")
    
    print(f"Analyzing {len(entities)} entities (starting from {start_idx})...")
    
    for i, entity in enumerate(tqdm(entities[start_idx:], initial=start_idx, total=len(entities))):
        try:
            neighbors = model.most_similar(entity, count=50)
            for neighbor_item, similarity in neighbors:
                if not hasattr(neighbor_item, 'title'):
                    continue
                if similarity < similarity_threshold:
                    break
                
                pair_key = tuple(sorted([entity.title.lower(), neighbor_item.title.lower()]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                if not are_linked(entity, neighbor_item, link_set):
                    candidates.append({
                        'entity_a': entity.title,
                        'entity_b': neighbor_item.title,
                        'similarity': float(similarity),
                        'linked': False
                    })
                    if len(candidates) >= max_pairs:
                        break
        except Exception:
            continue
        
        # Periodic checkpoint
        actual_idx = start_idx + i + 1
        if actual_idx % SAVE_INTERVAL == 0:
            print(f"\n[CHECKPOINT] Saving at entity {actual_idx}/{len(entities)}, {len(candidates)} candidates found...")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'candidates': candidates, 'seen_pairs': seen_pairs, 'next_idx': actual_idx}, f)
        
        if len(candidates) >= max_pairs:
            break
    
    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Save final output
    with open(output_path, 'w') as f:
        json.dump(candidates, f, indent=2)
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    return candidates

if __name__ == "__main__":
    data_dir = os.path.expanduser("~/data")
    results_dir = os.path.expanduser("~/data/results")
    checkpoint_dir = os.path.expanduser("~/data/.checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 2015 Analysis
    done_file = f"{checkpoint_dir}/hidden_2015.done"
    if not os.path.exists(done_file):
        print("Loading 2015 model...")
        model_2015 = Wikipedia2Vec.load(f"{data_dir}/models/wiki2015.pkl")
        print("Loading 2015 links...")
        links_2015 = load_link_set(f"{data_dir}/links_2015.json")
        
        print("Finding hidden connections in 2015 embeddings...")
        candidates_2015 = find_close_but_unlinked(
            model_2015, links_2015,
            f"{results_dir}/candidates_2015.json",
            f"{checkpoint_dir}/hidden_2015.progress"
        )
        print(f"\nFound {len(candidates_2015)} candidate hidden connections")
        print(f"\nTop 20 predictions:")
        for c in candidates_2015[:20]:
            print(f"  {c['entity_a']} <--> {c['entity_b']}  (similarity: {c['similarity']:.3f})")
        open(done_file, 'w').close()
        del model_2015, links_2015  # Free memory
    else:
        print("Skipping 2015 analysis (already done)")
    
    # 2025 Analysis
    done_file = f"{checkpoint_dir}/hidden_2025.done"
    if not os.path.exists(done_file):
        print("\nLoading 2025 model...")
        model_2025 = Wikipedia2Vec.load(f"{data_dir}/models/wiki2025.pkl")
        print("Loading 2025 links...")
        links_2025 = load_link_set(f"{data_dir}/links_2025.json")
        
        print("Finding hidden connections in 2025 embeddings...")
        candidates_2025 = find_close_but_unlinked(
            model_2025, links_2025,
            f"{results_dir}/candidates_2025.json",
            f"{checkpoint_dir}/hidden_2025.progress"
        )
        print(f"\nFound {len(candidates_2025)} candidate hidden connections (FUTURE)")
        print(f"\nTop 20 predictions:")
        for c in candidates_2025[:20]:
            print(f"  {c['entity_a']} <--> {c['entity_b']}  (similarity: {c['similarity']:.3f})")
        open(done_file, 'w').close()
    else:
        print("Skipping 2025 analysis (already done)")
