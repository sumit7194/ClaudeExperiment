"""Single source of truth for all experiment parameters.
Provides reproducibility via fixed seeds, versioned config, and run manifests."""

import json
import os
import platform
import sys
from datetime import datetime

EXPERIMENT_CONFIG = {
    # Metadata
    'experiment_name': 'temporal-knowledge-gap-finder',
    'run_id': None,  # Auto-generated at runtime
    'random_seed': 42,

    # Data sources
    'dumps': {
        '2015': {
            'url': 'https://dumps.wikimedia.org/enwiki/20150901/enwiki-20150901-pages-articles.xml.bz2',
            'sha256': None,
        },
        '2025': {
            'url': 'https://dumps.wikimedia.org/enwiki/20251220/enwiki-20251220-pages-articles.xml.bz2',
            'sha256': None,
        },
    },

    # Training hyperparameters (identical for both years)
    'training': {
        'dim_size': 300,
        'window': 10,
        'iteration': 10,
        'negative': 15,
        'min_entity_count': 5,
        'lowercase': True,
        'pool_size': 4,
    },

    # Analysis parameters
    'analysis': {
        'sample_size': 30000,
        'top_k_neighbors': 50,
        'n_clusters': 20,
        'shortest_path_max_depth': 4,
        'shortest_path_top_n': 1000,
        'precision_k_values': [10, 50, 100, 500, 1000, 5000, 10000],
        'similarity_thresholds': [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        'random_baseline_trials': 20,
        'random_baseline_samples': 10000,
        'min_entity_name_length': 3,
        'save_interval_entities': 1000,
        'save_interval_articles': 50000,
    },

    # Confidence scoring weights
    'confidence_weights': {
        'cosine_similarity': 0.40,
        'neighbor_overlap': 0.25,
        'cross_domain': 0.15,
        'entity_frequency_sweetspot': 0.20,
    },
}

# Directories
DATA_DIR = os.path.expanduser("~/data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
CHECKPOINT_DIR = os.path.join(DATA_DIR, ".checkpoints")


def init_dirs():
    """Create all required directories."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_run_id():
    """Generate or retrieve the current run ID."""
    if EXPERIMENT_CONFIG['run_id'] is None:
        EXPERIMENT_CONFIG['run_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    return EXPERIMENT_CONFIG['run_id']


def set_all_seeds():
    """Set random seeds for reproducibility across all libraries."""
    import random
    import numpy as np
    seed = EXPERIMENT_CONFIG['random_seed']
    random.seed(seed)
    np.random.seed(seed)


def save_run_manifest(output_dir=None):
    """Save complete run manifest for reproducibility."""
    if output_dir is None:
        output_dir = RESULTS_DIR

    try:
        import pkg_resources
        packages = {
            pkg.key: pkg.version
            for pkg in pkg_resources.working_set
            if pkg.key in ['wikipedia2vec', 'numpy', 'scipy', 'scikit-learn',
                           'gensim', 'networkx', 'matplotlib', 'tqdm']
        }
    except Exception:
        packages = {}

    manifest = {
        'config': EXPERIMENT_CONFIG,
        'environment': {
            'python_version': sys.version,
            'platform': platform.platform(),
            'cpu_count': os.cpu_count(),
            'packages': packages,
        },
        'timestamp_start': datetime.now().isoformat(),
    }

    manifest_path = os.path.join(output_dir, "run_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"Run manifest saved: {manifest_path}")
    return manifest


def get_rss_mb():
    """Get current process RSS memory in MB."""
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes, Linux returns KB
    if sys.platform == 'darwin':
        return usage / (1024 * 1024)
    return usage / 1024


def log_memory(label=""):
    """Log current memory usage with an optional label."""
    print(f"[MEMORY] {label} RSS: {get_rss_mb():.0f} MB")
