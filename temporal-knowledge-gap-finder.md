# The Temporal Knowledge Gap Finder

## A Complete Experiment Guide

**Goal:** Use Wikipedia embeddings from different time periods to discover that vector proximity between concepts can predict future knowledge connections — and then use current embeddings to predict connections that haven't been made yet.

**Inspired by:** Tshitoyan et al. (2019), "Unsupervised word embeddings capture latent knowledge from materials science literature" (Nature), which predicted thermoelectric materials years before their discovery using only text embeddings.

---

## The Core Idea

Wikipedia is a living record of human knowledge. When two concepts appear in similar contexts across Wikipedia articles, embedding models place them close together in vector space — even if no article explicitly connects them. The hypothesis is:

> **If two concepts are close in embedding space but have no direct Wikipedia link between them, that "hidden" proximity may predict a future explicit connection.**

By comparing embeddings from 2015 with what actually happened in Wikipedia by 2025, we can test this hypothesis. Then we use 2025 embeddings to make new predictions.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    PHASE 1: HISTORICAL                   │
│                                                          │
│  Wikipedia 2015 Dump ──► Train Embeddings (2015)         │
│         │                       │                        │
│         │                       ▼                        │
│         │              Find "close but unlinked"         │
│         │              concept pairs in 2015             │
│         │                       │                        │
│         ▼                       ▼                        │
│  Wikipedia 2025 Dump ──► Check: Did links appear         │
│                          between 2015-2025?              │
│                                 │                        │
│                                 ▼                        │
│                          VALIDATION SCORE                │
│                          (Precision / Recall)            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    PHASE 2: PREDICTION                   │
│                                                          │
│  Wikipedia 2025 Dump ──► Train Embeddings (2025)         │
│                                 │                        │
│                                 ▼                        │
│                        Find "close but unlinked"         │
│                        concept pairs in 2025             │
│                                 │                        │
│                                 ▼                        │
│                        PREDICTIONS LIST                  │
│                        (Future connections)              │
└─────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Hardware
- **Minimum:** 16 GB RAM, 100 GB free disk space, any modern CPU
- **Recommended:** 32 GB RAM, 200 GB disk, multi-core CPU (8+ cores)
- **GPU:** Not required (Wikipedia2Vec is CPU-optimized with BLAS)
- **Time estimate:** ~4-8 hours for training embeddings on full English Wikipedia per dump

### Software Stack

```bash
# Python 3.9+
python --version

# Core dependencies
pip install wikipedia2vec gensim numpy scipy scikit-learn pandas matplotlib tqdm

# For Wikipedia dump processing
pip install mwparserfromhell bz2file

# For graph/link analysis
pip install networkx

# Optional: for interactive exploration
pip install plotly umap-learn jupyter
```

### Disk Space Budget

| Item | Size (approx.) |
|------|----------------|
| Wikipedia 2015 dump (compressed) | ~14 GB |
| Wikipedia 2025 dump (compressed) | ~22 GB |
| Processed dump DB (per year) | ~8 GB |
| Trained embeddings (per year) | ~3-6 GB |
| Working space | ~10 GB |
| **Total** | **~70-80 GB** |

---

## Phase 0: Data Acquisition

### Step 0.1 — Download Historical Wikipedia Dumps

Wikipedia archives all past dumps. You need dumps from two time periods.

```bash
mkdir -p data/dumps && cd data/dumps

# 2015 dump (archived)
# Browse available dates at: https://dumps.wikimedia.org/enwiki/
# Archived dumps: https://archive.org/details/enwiki-20150901
wget https://dumps.wikimedia.org/enwiki/20150901/enwiki-20150901-pages-articles.xml.bz2

# 2025 dump (recent)
wget https://dumps.wikimedia.org/enwiki/20250101/enwiki-20250101-pages-articles.xml.bz2
```

**Important notes:**
- Older dumps may not be on Wikimedia's servers. Check the Internet Archive: `https://archive.org/search?query=enwiki+pages-articles`
- If exact dates aren't available, the nearest available date works fine. The key is having a ~10 year gap.
- You do NOT need to decompress the `.bz2` files — Wikipedia2Vec reads them directly.

### Step 0.2 — Verify Downloads

```bash
# Check file sizes (should be 10+ GB each)
ls -lh data/dumps/

# Verify bz2 integrity
bzip2 -t enwiki-20150901-pages-articles.xml.bz2
bzip2 -t enwiki-20250101-pages-articles.xml.bz2
```

---

## Phase 1: Training Embeddings

### Step 1.1 — Train 2015 Embeddings

```bash
# This single command handles everything:
# - Parses the XML dump
# - Builds word and entity dictionaries
# - Extracts link graph
# - Trains embeddings

wikipedia2vec train \
  data/dumps/enwiki-20150901-pages-articles.xml.bz2 \
  models/wiki2015.pkl \
  --dim-size 300 \
  --window 10 \
  --iteration 10 \
  --negative 15 \
  --min-entity-count 5 \
  --lowercase \
  --pool-size 8           # adjust to your CPU core count
```

**Parameter choices explained:**
- `--dim-size 300`: Standard dimensionality. Good balance of expressiveness and compute.
- `--window 10`: Larger window captures broader topical associations (important for cross-concept links).
- `--iteration 10`: More iterations = more stable embeddings. 5 is minimum, 10 is solid.
- `--min-entity-count 5`: Skip very obscure entities (reduces noise).
- `--negative 15`: Negative sampling rate. Higher = better quality but slower.

### Step 1.2 — Train 2025 Embeddings

```bash
wikipedia2vec train \
  data/dumps/enwiki-20250101-pages-articles.xml.bz2 \
  models/wiki2025.pkl \
  --dim-size 300 \
  --window 10 \
  --iteration 10 \
  --negative 15 \
  --min-entity-count 5 \
  --lowercase \
  --pool-size 8
```

**Critical: Use identical hyperparameters** for both years so the embedding spaces are comparable.

### Step 1.3 — Quick Sanity Check

```python
"""sanity_check.py — Verify embeddings loaded correctly."""

from wikipedia2vec import Wikipedia2Vec

# Load models
model_2015 = Wikipedia2Vec.load("models/wiki2015.pkl")
model_2025 = Wikipedia2Vec.load("models/wiki2025.pkl")

# Basic stats
print(f"2015 model: {len(model_2015.dictionary.entities())} entities, "
      f"{len(model_2015.dictionary.words())} words")
print(f"2025 model: {len(model_2025.dictionary.entities())} entities, "
      f"{len(model_2025.dictionary.words())} words")

# Sanity: check known relationships
for title in ["Albert Einstein", "Python (programming language)", "Photosynthesis"]:
    entity = model_2025.get_entity(title)
    if entity:
        similar = model_2025.most_similar(entity, count=5)
        print(f"\nMost similar to '{title}':")
        for item, score in similar:
            print(f"  {item} ({score:.3f})")
```

Expected output: entities similar to "Albert Einstein" should include physics-related concepts; "Python (programming language)" should show other programming languages or CS concepts.

---

## Phase 2: Extract Link Graphs

This is where we capture the *explicit* connections in Wikipedia for each year.

### Step 2.1 — Build Link Graph from Dumps

```python
"""extract_links.py — Extract article-to-article links from Wikipedia dumps."""

import json
import re
from collections import defaultdict
from wikipedia2vec.dump_db import DumpDB
from tqdm import tqdm

def extract_link_graph(dump_db_path, output_path):
    """
    Extract all internal links between Wikipedia articles.
    
    Returns a dict: {source_title: set(target_titles)}
    """
    dump_db = DumpDB(dump_db_path)
    link_graph = defaultdict(set)
    
    titles = list(dump_db.titles())
    print(f"Processing {len(titles)} articles...")
    
    for title in tqdm(titles):
        try:
            # Get all internal links from this article
            for link in dump_db.get_paragraphs(title):
                for wiki_link in link.wiki_links:
                    target = wiki_link.title
                    if target:
                        link_graph[title].add(target)
        except Exception as e:
            continue
    
    # Save as JSON (convert sets to lists for serialization)
    serializable = {k: list(v) for k, v in link_graph.items()}
    with open(output_path, 'w') as f:
        json.dump(serializable, f)
    
    print(f"Extracted links from {len(link_graph)} articles")
    print(f"Total unique links: {sum(len(v) for v in link_graph.values())}")
    
    return link_graph


# NOTE: You need to build DumpDB first if not done during training.
# wikipedia2vec builds this internally, but you can also do:
#   wikipedia2vec build-dump-db <dump.xml.bz2> <output.pkl>

# Build link graphs for both years
# (adjust paths to your DumpDB files)
links_2015 = extract_link_graph("data/dumpdb_2015.pkl", "data/links_2015.json")
links_2025 = extract_link_graph("data/dumpdb_2025.pkl", "data/links_2025.json")
```

### Step 2.2 — Alternative: Lightweight Link Extraction via SQL Dumps

If the above is too slow, Wikipedia also provides pre-built link tables:

```bash
# Download the pagelinks SQL dump (much smaller than full article dump)
wget https://dumps.wikimedia.org/enwiki/20150901/enwiki-20150901-pagelinks.sql.gz
wget https://dumps.wikimedia.org/enwiki/20250101/enwiki-20250101-pagelinks.sql.gz
```

```python
"""parse_pagelinks.py — Parse SQL link dumps (faster alternative)."""

import gzip
import re
from collections import defaultdict

def parse_pagelinks_sql(sql_gz_path):
    """Parse pagelinks SQL dump into a link dictionary."""
    links = defaultdict(set)
    pattern = re.compile(r"\((\d+),(\d+),'(.+?)',(\d+)\)")
    
    with gzip.open(sql_gz_path, 'rt', errors='replace') as f:
        for line in f:
            if not line.startswith("INSERT INTO"):
                continue
            for match in pattern.finditer(line):
                source_id = int(match.group(1))
                target_ns = int(match.group(2))
                target_title = match.group(3).replace('_', ' ')
                # namespace 0 = main articles only
                if target_ns == 0:
                    links[source_id].add(target_title)
    
    return links
```

---

## Phase 3: Find "Close but Unlinked" Concept Pairs (2015)

This is the heart of the experiment.

### Step 3.1 — Compute Concept Proximity

```python
"""find_hidden_connections.py — Core experiment logic."""

import numpy as np
from wikipedia2vec import Wikipedia2Vec
from scipy.spatial.distance import cosine
from collections import defaultdict
import json
from tqdm import tqdm
import csv

def load_link_set(links_path):
    """Load link graph as a set of (source, target) pairs for fast lookup."""
    with open(links_path) as f:
        link_dict = json.load(f)
    
    link_set = set()
    for source, targets in link_dict.items():
        for target in targets:
            link_set.add((source.lower(), target.lower()))
            # Add reverse direction too (undirected check)
            link_set.add((target.lower(), source.lower()))
    
    return link_set


def are_linked(entity_a, entity_b, link_set):
    """Check if two entities have a direct Wikipedia link between them."""
    a = entity_a.title.lower()
    b = entity_b.title.lower()
    return (a, b) in link_set or (b, a) in link_set


def find_close_but_unlinked(model, link_set, 
                             similarity_threshold=0.65,
                             max_pairs=50000,
                             min_entity_name_length=3):
    """
    Find entity pairs that are close in embedding space 
    but NOT linked in Wikipedia.
    
    These are our "hidden connection" candidates.
    """
    entities = [e for e in model.dictionary.entities() 
                if len(e.title) >= min_entity_name_length]
    
    print(f"Analyzing {len(entities)} entities...")
    
    # Strategy: For each entity, find its nearest neighbors
    # and check if they're linked. Unlinked close neighbors = candidates.
    candidates = []
    
    for entity in tqdm(entities[:20000]):  # Limit for tractability
        try:
            # Get top 50 most similar entities
            neighbors = model.most_similar(entity, count=50)
            
            for neighbor_item, similarity in neighbors:
                # Skip word results (we only want entity-entity pairs)
                if not hasattr(neighbor_item, 'title'):
                    continue
                
                neighbor = neighbor_item
                
                if similarity < similarity_threshold:
                    break  # Already sorted by similarity
                
                # THE KEY CHECK: close in embedding space but NOT linked
                if not are_linked(entity, neighbor, link_set):
                    candidates.append({
                        'entity_a': entity.title,
                        'entity_b': neighbor.title,
                        'similarity': float(similarity),
                        'linked_2015': False
                    })
                    
                    if len(candidates) >= max_pairs:
                        return candidates
        except Exception:
            continue
    
    # Sort by similarity (highest first = strongest predictions)
    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    
    return candidates


# ── Run the analysis ─────────────────────────────────────

print("Loading 2015 model...")
model_2015 = Wikipedia2Vec.load("models/wiki2015.pkl")

print("Loading 2015 links...")
links_2015 = load_link_set("data/links_2015.json")

print("Finding hidden connections in 2015 embeddings...")
candidates_2015 = find_close_but_unlinked(
    model_2015, 
    links_2015,
    similarity_threshold=0.60  # Adjust based on your results
)

# Save candidates
with open("results/candidates_2015.json", 'w') as f:
    json.dump(candidates_2015, f, indent=2)

print(f"Found {len(candidates_2015)} candidate hidden connections")
print(f"\nTop 20 predictions:")
for c in candidates_2015[:20]:
    print(f"  {c['entity_a']} <──> {c['entity_b']}  "
          f"(similarity: {c['similarity']:.3f})")
```

### Step 3.2 — What the Output Looks Like

You should see pairs like:

```
CRISPR <──> Gene therapy  (similarity: 0.82)
Blockchain <──> Supply chain management  (similarity: 0.74)
Deep learning <──> Drug discovery  (similarity: 0.71)
...
```

These are concepts that "lived nearby" in 2015's knowledge space but weren't explicitly linked.

---

## Phase 4: Validate Against 2025 Reality

### Step 4.1 — Check Which Predictions Came True

```python
"""validate_predictions.py — The moment of truth."""

import json

def validate_predictions(candidates_2015, links_2025_path):
    """
    Check how many of our 2015 'hidden connections' became
    explicit Wikipedia links by 2025.
    """
    links_2025 = load_link_set(links_2025_path)
    
    total = len(candidates_2015)
    confirmed = 0
    confirmed_pairs = []
    unconfirmed_pairs = []
    
    for candidate in candidates_2015:
        a = candidate['entity_a'].lower()
        b = candidate['entity_b'].lower()
        
        now_linked = (a, b) in links_2025 or (b, a) in links_2025
        
        if now_linked:
            confirmed += 1
            candidate['confirmed'] = True
            confirmed_pairs.append(candidate)
        else:
            candidate['confirmed'] = False
            unconfirmed_pairs.append(candidate)
    
    return {
        'total_predictions': total,
        'confirmed': confirmed,
        'precision': confirmed / total if total > 0 else 0,
        'confirmed_pairs': confirmed_pairs,
        'unconfirmed_pairs': unconfirmed_pairs
    }


# ── Validate ─────────────────────────────────────────────

with open("results/candidates_2015.json") as f:
    candidates_2015 = json.load(f)

results = validate_predictions(candidates_2015, "data/links_2025.json")

print(f"═══════════════════════════════════════════════")
print(f"  VALIDATION RESULTS")
print(f"═══════════════════════════════════════════════")
print(f"  Total predictions from 2015:  {results['total_predictions']}")
print(f"  Confirmed by 2025:            {results['confirmed']}")
print(f"  Precision:                     {results['precision']:.1%}")
print(f"═══════════════════════════════════════════════")

print(f"\nTop confirmed connections (predicted in 2015, linked by 2025):")
for pair in results['confirmed_pairs'][:20]:
    print(f"  ✓ {pair['entity_a']} <──> {pair['entity_b']}  "
          f"(similarity: {pair['similarity']:.3f})")
```

### Step 4.2 — Precision-at-K Analysis

```python
"""precision_at_k.py — How prediction quality varies with confidence."""

import matplotlib.pyplot as plt

def precision_at_k(candidates, links_2025, k_values):
    """
    Calculate precision at different K values.
    Higher-similarity predictions should have higher precision.
    """
    # Sort by similarity (highest first)
    sorted_candidates = sorted(candidates, 
                                key=lambda x: x['similarity'], 
                                reverse=True)
    
    results = []
    for k in k_values:
        top_k = sorted_candidates[:k]
        confirmed = sum(1 for c in top_k 
                       if is_now_linked(c, links_2025))
        precision = confirmed / k
        results.append({'k': k, 'precision': precision})
        print(f"  Precision@{k:>6}: {precision:.3f} "
              f"({confirmed}/{k} confirmed)")
    
    return results


def is_now_linked(candidate, links_2025):
    a = candidate['entity_a'].lower()
    b = candidate['entity_b'].lower()
    return (a, b) in links_2025 or (b, a) in links_2025


# ── Compute Precision@K ──────────────────────────────────

links_2025 = load_link_set("data/links_2025.json")

k_values = [10, 50, 100, 500, 1000, 5000, 10000]
pak_results = precision_at_k(candidates_2015, links_2025, k_values)

# Plot
ks = [r['k'] for r in pak_results]
precisions = [r['precision'] for r in pak_results]

plt.figure(figsize=(10, 6))
plt.plot(ks, precisions, 'bo-', linewidth=2, markersize=8)
plt.xlabel('K (number of top predictions)', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision@K: Do Higher-Confidence Predictions Come True More Often?', 
          fontsize=14)
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/precision_at_k.png', dpi=150)
plt.show()
```

**What you hope to see:** A downward slope — the top 10 predictions should have much higher precision than the top 10,000. This means the embedding similarity is a real signal, not noise.

### Step 4.3 — Compare Against Random Baseline

```python
"""baseline_comparison.py — Is our method better than chance?"""

import random

def random_baseline_precision(all_entities_2015, links_2015, links_2025, 
                               n_samples=10000, n_trials=10):
    """
    Randomly pair unlinked entities and check how many 
    became linked. This is the 'chance' baseline.
    """
    entity_list = list(all_entities_2015)
    precisions = []
    
    for trial in range(n_trials):
        confirmed = 0
        for _ in range(n_samples):
            a = random.choice(entity_list)
            b = random.choice(entity_list)
            if a == b:
                continue
            
            a_lower, b_lower = a.lower(), b.lower()
            
            # Must be unlinked in 2015
            if (a_lower, b_lower) in links_2015:
                continue
            
            # Check if linked in 2025
            if (a_lower, b_lower) in links_2025 or (b_lower, a_lower) in links_2025:
                confirmed += 1
        
        precisions.append(confirmed / n_samples)
    
    avg = sum(precisions) / len(precisions)
    print(f"Random baseline precision: {avg:.4f} "
          f"(±{max(precisions)-min(precisions):.4f})")
    return avg
```

**What you hope to see:** Your embedding-based precision should be 10-100x higher than random. If embedding precision@100 is 15% and random is 0.5%, you have strong evidence the embeddings capture real latent knowledge.

---

## Phase 5: Generate New Predictions (2025 → Future)

### Step 5.1 — Run the Same Pipeline on 2025 Embeddings

```python
"""predict_future.py — The exciting part."""

print("Loading 2025 model...")
model_2025 = Wikipedia2Vec.load("models/wiki2025.pkl")

print("Loading 2025 links...")
links_2025 = load_link_set("data/links_2025.json")

print("Finding hidden connections in 2025 embeddings...")
future_candidates = find_close_but_unlinked(
    model_2025,
    links_2025,
    similarity_threshold=0.65
)

# Save predictions
with open("results/future_predictions_2025.json", 'w') as f:
    json.dump(future_candidates, f, indent=2)

print(f"\n{'═'*60}")
print(f"  FUTURE PREDICTIONS (2025 embeddings)")
print(f"  These concept pairs are close in embedding space")
print(f"  but not yet linked in Wikipedia.")
print(f"{'═'*60}")
print(f"\nTop 50 predicted future connections:\n")
for i, c in enumerate(future_candidates[:50], 1):
    print(f"  {i:>3}. {c['entity_a']}")
    print(f"       ↔ {c['entity_b']}")
    print(f"       similarity: {c['similarity']:.3f}")
    print()
```

### Step 5.2 — Categorize and Analyze Predictions

```python
"""categorize_predictions.py — Make predictions interpretable."""

def categorize_by_domain(predictions, model):
    """
    Group predictions by broad domain using embedding clusters.
    This helps identify which fields have the most 'hidden' connections.
    """
    # Define domain anchor concepts
    domains = {
        'Physics': ['Quantum mechanics', 'General relativity', 'Thermodynamics'],
        'Biology': ['DNA', 'Evolution', 'Cell biology'],
        'Computer Science': ['Algorithm', 'Machine learning', 'Computer network'],
        'Mathematics': ['Calculus', 'Linear algebra', 'Number theory'],
        'Chemistry': ['Chemical bond', 'Organic chemistry', 'Periodic table'],
        'Medicine': ['Disease', 'Pharmacology', 'Surgery'],
        'Economics': ['Macroeconomics', 'Supply and demand', 'Financial market'],
        'History': ['Ancient history', 'World War II', 'Industrial Revolution'],
    }
    
    # Compute domain centroids
    domain_vectors = {}
    for domain, anchors in domains.items():
        vectors = []
        for anchor in anchors:
            entity = model.get_entity(anchor)
            if entity:
                vectors.append(model.get_entity_vector(entity))
        if vectors:
            domain_vectors[domain] = np.mean(vectors, axis=0)
    
    # Classify each prediction
    for pred in predictions:
        entity_a = model.get_entity(pred['entity_a'])
        entity_b = model.get_entity(pred['entity_b'])
        
        if entity_a and entity_b:
            vec_a = model.get_entity_vector(entity_a)
            vec_b = model.get_entity_vector(entity_b)
            
            # Find closest domain for each entity
            domain_a = classify_domain(vec_a, domain_vectors)
            domain_b = classify_domain(vec_b, domain_vectors)
            
            pred['domain_a'] = domain_a
            pred['domain_b'] = domain_b
            pred['cross_domain'] = (domain_a != domain_b)
    
    return predictions


def classify_domain(vector, domain_vectors):
    """Assign a vector to its closest domain."""
    best_domain = None
    best_sim = -1
    for domain, centroid in domain_vectors.items():
        sim = np.dot(vector, centroid) / (
            np.linalg.norm(vector) * np.linalg.norm(centroid))
        if sim > best_sim:
            best_sim = sim
            best_domain = domain
    return best_domain
```

**The most interesting predictions are cross-domain ones** — these suggest connections between fields that haven't been explicitly recognized yet.

---

## Phase 6: Visualization and Analysis

### Step 6.1 — Embedding Space Visualization

```python
"""visualize_embeddings.py — See the knowledge landscape."""

import numpy as np
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go

def visualize_predictions(model, predictions, n_context=500):
    """
    Visualize predicted connections in 2D embedding space.
    Shows the concept landscape with predicted links highlighted.
    """
    # Collect entities: predictions + context entities
    entities_to_plot = set()
    
    for pred in predictions[:100]:  # Top 100 predictions
        entities_to_plot.add(pred['entity_a'])
        entities_to_plot.add(pred['entity_b'])
    
    # Add random context entities for landscape
    all_entities = list(model.dictionary.entities())
    random_sample = np.random.choice(
        len(all_entities), 
        size=min(n_context, len(all_entities)), 
        replace=False
    )
    for idx in random_sample:
        entities_to_plot.add(all_entities[idx].title)
    
    # Build vector matrix
    titles = []
    vectors = []
    is_prediction = []
    
    for title in entities_to_plot:
        entity = model.get_entity(title)
        if entity:
            titles.append(title)
            vectors.append(model.get_entity_vector(entity))
            is_prediction.append(
                title in {p['entity_a'] for p in predictions[:100]} |
                        {p['entity_b'] for p in predictions[:100]}
            )
    
    vectors = np.array(vectors)
    
    # Reduce to 2D
    reducer = UMAP(n_neighbors=30, min_dist=0.1, metric='cosine')
    coords = reducer.fit_transform(vectors)
    
    # Plot
    fig = px.scatter(
        x=coords[:, 0], y=coords[:, 1],
        text=titles,
        color=['Predicted Connection' if p else 'Context' for p in is_prediction],
        color_discrete_map={
            'Predicted Connection': '#FF6B6B',
            'Context': '#C0C0C0'
        },
        title='Wikipedia Knowledge Landscape with Predicted Future Connections',
        width=1200, height=800
    )
    
    # Draw lines between predicted pairs
    for pred in predictions[:50]:
        a_idx = titles.index(pred['entity_a']) if pred['entity_a'] in titles else None
        b_idx = titles.index(pred['entity_b']) if pred['entity_b'] in titles else None
        
        if a_idx is not None and b_idx is not None:
            fig.add_trace(go.Scatter(
                x=[coords[a_idx, 0], coords[b_idx, 0]],
                y=[coords[a_idx, 1], coords[b_idx, 1]],
                mode='lines',
                line=dict(color='rgba(255,100,100,0.3)', width=1),
                showlegend=False
            ))
    
    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
    fig.write_html('results/knowledge_landscape.html')
    fig.show()
```

### Step 6.2 — Temporal Drift Visualization

```python
"""temporal_drift.py — How has the knowledge landscape shifted?"""

def compare_embedding_neighborhoods(model_old, model_new, entity_title, k=20):
    """
    Show how a concept's neighborhood changed between time periods.
    Concepts that gained new neighbors may represent emerging connections.
    """
    entity_old = model_old.get_entity(entity_title)
    entity_new = model_new.get_entity(entity_title)
    
    if not entity_old or not entity_new:
        print(f"Entity '{entity_title}' not found in both models.")
        return
    
    neighbors_old = set()
    for item, score in model_old.most_similar(entity_old, count=k):
        if hasattr(item, 'title'):
            neighbors_old.add(item.title)
    
    neighbors_new = set()
    for item, score in model_new.most_similar(entity_new, count=k):
        if hasattr(item, 'title'):
            neighbors_new.add(item.title)
    
    gained = neighbors_new - neighbors_old
    lost = neighbors_old - neighbors_new
    stable = neighbors_old & neighbors_new
    
    print(f"\n{'═'*50}")
    print(f"  Neighborhood shift: {entity_title}")
    print(f"{'═'*50}")
    print(f"\n  Stable neighbors ({len(stable)}):")
    for n in list(stable)[:10]:
        print(f"    • {n}")
    print(f"\n  NEW neighbors in 2025 ({len(gained)}):")
    for n in gained:
        print(f"    + {n}")
    print(f"\n  LOST neighbors from 2015 ({len(lost)}):")
    for n in lost:
        print(f"    - {n}")


# ── Explore interesting concepts ─────────────────────────

concepts_to_track = [
    "CRISPR",
    "Artificial intelligence", 
    "Climate change",
    "Blockchain",
    "Quantum computing",
    "Microbiome",
]

for concept in concepts_to_track:
    compare_embedding_neighborhoods(model_2015, model_2025, concept)
```

---

## Phase 7: Advanced Experiments

Once the basic pipeline works, try these extensions.

### 7.1 — Domain-Specific Deep Dives

Instead of all of Wikipedia, filter to specific categories:

```python
def get_category_articles(dump_db, category_prefix):
    """Extract articles belonging to a specific category tree."""
    # Use Wikipedia's category system to filter
    # e.g., all articles in "Category:Mathematics" and subcategories
    pass

# Train separate embeddings for:
# - All math articles only
# - All physics articles only  
# - All biology articles only
# Then look for cross-domain hidden connections
```

### 7.2 — Analogy-Based Prediction

Go beyond simple proximity — use vector arithmetic:

```python
def analogy_prediction(model, a, b, c):
    """
    If A is to B as C is to ???
    Computes: vec(B) - vec(A) + vec(C) = ???
    """
    entity_a = model.get_entity(a)
    entity_b = model.get_entity(b)
    entity_c = model.get_entity(c)
    
    if not all([entity_a, entity_b, entity_c]):
        return None
    
    vec_a = model.get_entity_vector(entity_a)
    vec_b = model.get_entity_vector(entity_b)
    vec_c = model.get_entity_vector(entity_c)
    
    # The analogy vector
    target_vec = vec_b - vec_a + vec_c
    
    # Find nearest entities to the target vector
    # (Wikipedia2Vec doesn't have a direct method for this,
    #  so we compute similarities manually)
    best_matches = []
    for entity in model.dictionary.entities():
        vec = model.get_entity_vector(entity)
        sim = np.dot(target_vec, vec) / (
            np.linalg.norm(target_vec) * np.linalg.norm(vec))
        best_matches.append((entity.title, float(sim)))
    
    best_matches.sort(key=lambda x: x[1], reverse=True)
    
    # Filter out the input entities
    exclude = {a, b, c}
    return [(title, sim) for title, sim in best_matches 
            if title not in exclude][:10]


# Example: What is to biology what deep learning is to computer science?
results = analogy_prediction(
    model_2025,
    "Computer science",  # A
    "Deep learning",     # B  
    "Biology"            # C
)
print("Computer Science : Deep Learning :: Biology : ???")
for title, sim in results:
    print(f"  {title} ({sim:.3f})")
```

### 7.3 — Confidence Calibration

Build a scoring function that combines multiple signals:

```python
def prediction_confidence(entity_a, entity_b, model, link_set):
    """
    Multi-signal confidence score for a predicted connection.
    """
    e_a = model.get_entity(entity_a)
    e_b = model.get_entity(entity_b)
    
    if not e_a or not e_b:
        return 0.0
    
    vec_a = model.get_entity_vector(e_a)
    vec_b = model.get_entity_vector(e_b)
    
    # Signal 1: Direct cosine similarity
    direct_sim = np.dot(vec_a, vec_b) / (
        np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    
    # Signal 2: Shared neighbor overlap
    neighbors_a = {item.title for item, _ in model.most_similar(e_a, count=50)
                   if hasattr(item, 'title')}
    neighbors_b = {item.title for item, _ in model.most_similar(e_b, count=50)
                   if hasattr(item, 'title')}
    neighbor_overlap = len(neighbors_a & neighbors_b) / max(
        len(neighbors_a | neighbors_b), 1)
    
    # Signal 3: Both entities are "important" (have many links)
    links_a = sum(1 for k, v in link_set if k == entity_a.lower())
    links_b = sum(1 for k, v in link_set if k == entity_b.lower())
    importance = min(links_a, links_b) / max(links_a, links_b, 1)
    
    # Combined score (weights can be tuned)
    confidence = (0.5 * direct_sim + 
                  0.3 * neighbor_overlap + 
                  0.2 * importance)
    
    return confidence
```

---

## Expected Results and Interpretation

### What "Good" Results Look Like

| Metric | Weak Signal | Moderate Signal | Strong Signal |
|--------|-------------|-----------------|---------------|
| Precision@100 | 5-10% | 10-25% | 25%+ |
| Precision@1000 | 2-5% | 5-15% | 15%+ |
| vs. Random Baseline | 2-5x | 5-20x | 20x+ |

Even a "weak signal" result is scientifically interesting — it means text co-occurrence patterns contain genuine predictive information about future knowledge connections.

### What the Predictions Mean

The predictions fall into roughly three categories:

1. **Obvious-in-hindsight connections** — These are pairs where an expert would say "of course those should be linked." The embedding caught what editors hadn't gotten around to documenting yet. Still useful (could drive Wikipedia improvement bots).

2. **Emerging cross-domain connections** — The most exciting category. These suggest that two fields are converging but the explicit connection hasn't been recognized. Example: if "Transformer (machine learning)" and "Protein folding" were close in 2018 embeddings, that would have predicted AlphaFold's approach.

3. **Artifacts / false positives** — Some pairs will be close because of superficial textual similarity (shared jargon, similar article structure) rather than deep conceptual connection. Filtering these is an ongoing challenge.

### How to Judge Your Predictions

For the future predictions (2025 → ???), you can't validate automatically yet. Instead:

- **Manual review:** Read the top 50 predictions. Do they make intuitive sense? Could you write a paragraph explaining why these concepts might be connected?
- **Expert review:** Share predictions with domain experts. The materials science paper found that their top predictions included materials that were independently being studied.
- **Wait and check:** Re-run the validation in 2-3 years with a newer dump.

---

## Scaling Up: Going Bigger

### Use Pre-trained Embeddings (Skip Training)

If you want to skip the training step entirely for a quick start:

```python
# Download pre-trained Wikipedia2Vec embeddings
# Available at: https://wikipedia2vec.github.io/wikipedia2vec/pretrained/
# English, 300d, trained on full Wikipedia

from wikipedia2vec import Wikipedia2Vec
model = Wikipedia2Vec.load("enwiki_20180420_300d.pkl.bz2")
```

The trade-off: you only get one time point, so you can't do temporal validation. But you can still find "close but unlinked" pairs and manually assess them.

### Combine with Wikidata

Wikidata has structured relationships (not just links). Check whether your predicted pairs have Wikidata connections, which are more curated than Wikipedia hyperlinks:

```bash
# Download Wikidata JSON dump
wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2
```

### Run on Non-English Wikipedias

Train embeddings on Hindi, Japanese, and French Wikipedia. Compare predictions across languages. Where do predictions agree across languages? Those are likely the strongest.

---

## Project Deliverables Checklist

When you're done, you should have:

- [ ] Trained embedding models for 2 time periods
- [ ] Link graphs for both periods
- [ ] List of "hidden connections" from 2015 embeddings
- [ ] Validation results showing how many became real by 2025
- [ ] Precision@K curve showing prediction quality
- [ ] Random baseline comparison
- [ ] List of future predictions from 2025 embeddings
- [ ] Visualization of the knowledge landscape
- [ ] Neighborhood drift analysis for key concepts
- [ ] A writeup interpreting the most interesting findings

---

## Quick-Start Shortcut (Get Results in 1 Day)

If you want results fast without training:

1. Download pre-trained Wikipedia2Vec embeddings (any year available)
2. Download the corresponding Wikipedia pagelinks SQL dump
3. Skip Phase 1 training, go directly to Phase 3
4. Find "close but unlinked" pairs
5. Manually review the top 100 for interestingness

This gives you the "prediction" half without the temporal validation, but you'll immediately see whether the approach surfaces interesting hidden connections.

---

*Last updated: March 2026*
*Author: Sumit & Claude collaboration*
