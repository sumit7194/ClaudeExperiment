---
name: gcp-pipeline
description: >
  Manage long-running ML/data pipelines on GCP VMs with checkpoint/resume logic,
  progress monitoring, and cost management. Use this skill whenever working with
  remote compute jobs, Wikipedia2Vec training, embedding pipelines, or any
  multi-hour process on cloud VMs that needs fault tolerance. Triggers on:
  pipeline monitoring, checkpoint recovery, GCP VM management, resume interrupted
  training, background job status, compute cost tracking.
---

# GCP Pipeline Manager

A skill for managing long-running ML/data pipelines on GCP VMs with built-in
fault tolerance, checkpoint/resume capabilities, and progress monitoring.

## Core Principles

### 1. Never Lose Progress
Every pipeline step that takes more than 5 minutes should:
- Check for existing checkpoint before starting (skip if done)
- Save intermediate results periodically during execution
- Write a completion marker when finished
- Log wall-clock time for each step

### 2. Checkpoint Pattern
```bash
CHECKPOINT_DIR="$HOME/data/.checkpoints"
mkdir -p "$CHECKPOINT_DIR"

mark_done() { touch "$CHECKPOINT_DIR/$1.done"; }
is_done() { [ -f "$CHECKPOINT_DIR/$1.done" ]; }

# Usage in scripts:
if is_done "step_name"; then
    echo "Skipping step_name (already completed)"
else
    # ... do work ...
    mark_done "step_name"
fi
```

### 3. Periodic Saving for Long Loops
For Python scripts iterating over large datasets:
```python
# Save every N iterations
if i % SAVE_INTERVAL == 0:
    save_checkpoint(results, checkpoint_path)
    
# On startup, load from checkpoint if exists
if os.path.exists(checkpoint_path):
    results, start_idx = load_checkpoint(checkpoint_path)
```

### 4. Wikipedia2Vec Specific
The `wikipedia2vec train` command is monolithic — split into subcommands:
1. `build-dump-db` — Parse XML to database (~1-2h)
2. `build-dictionary` — Extract vocabulary (~30m)  
3. `build-link-graph` — Extract hyperlinks (~30m)
4. `build-mention-db` — Build mention index (~30m)
5. `train-embedding` — The actual training (~4-8h)

Each subcommand can be checkpointed independently.

### 5. VM Cost Management
- Use `e2-standard-4` (4 vCPU, 16GB) for CPU-bound work — ~$0.13/hr
- Stop VM when not in use: `gcloud compute instances stop <vm>`
- Monitor spend: pipeline should log estimated cost based on runtime
- Set budget alerts in GCP console

### 6. Monitoring
- Dashboard on port 8080 with auto-refresh
- Pipeline log at `~/pipeline.log`
- Check progress: `tail -20 ~/pipeline.log`
- Check system: `htop`, `df -h`, `free -h`

## File Layout on VM
```
~/data/
  dumps/          # Raw Wikipedia dumps (.bz2)
  dumpdb_*.pkl    # Parsed dump databases
  dict_*.pkl      # Dictionaries
  link_graph_*.pkl # Link graphs
  mention_db_*.pkl # Mention databases
  models/         # Trained embeddings
  results/        # Experiment outputs
  .checkpoints/   # Completion markers
~/ClaudeExperiment/
  scripts/        # Pipeline scripts
  skills/         # This skill
  results/        # Committed results (git tracked)
```
