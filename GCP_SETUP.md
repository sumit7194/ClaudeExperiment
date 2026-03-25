# GCP VM Setup — Temporal Knowledge Gap Finder

## Project Details

| Field | Value |
|-------|-------|
| **Project Name** | Temporal-knowledge-gap-finder |
| **Project ID** | `temporal-knowledge-gap-finder` |
| **Project Number** | 423894603805 |
| **Credits** | ₹27,287 (expires 24 June 2026) |
| **Region** | asia-south1 (Mumbai, India) |
| **Zone** | asia-south1-a |

## VM Details

| Field | Value |
|-------|-------|
| **VM Name** | `wiki-embeddings-vm` |
| **Machine Type** | e2-standard-4 (4 vCPU, 16 GB RAM) |
| **Disk** | 200 GB Standard Persistent Disk |
| **OS** | Ubuntu (default GCP image) |
| **External IP** | 34.100.208.184 |
| **Estimated Cost** | ~$0.13/hr (~$3.12/day if running 24/7) |

## Dashboard

**URL:** http://34.100.208.184:8080

- Auto-refreshes every 30 seconds
- Shows pipeline step status, system stats, and live log
- Firewall rule `allow-dashboard` opens TCP port 8080

## How to Connect

### SSH via gcloud CLI
```bash
gcloud compute ssh wiki-embeddings-vm --zone=asia-south1-a
```

### Check Pipeline Progress
```bash
# From your local machine (no SSH needed):
gcloud compute ssh wiki-embeddings-vm --zone=asia-south1-a --command="tail -20 ~/pipeline.log"

# Or once SSH'd in:
tail -20 ~/pipeline.log
bash ~/check_downloads.sh
```

### Check Checkpoint Status
```bash
source ~/ClaudeExperiment/scripts/checkpoint.sh && show_checkpoints
```

### View System Resources
```bash
htop          # CPU and memory
df -h /home   # Disk space
```

## Key Paths on the VM

```
~/data/
  dumps/                  # Wikipedia dump files (~36 GB)
  dumpdb_2015.pkl         # Parsed 2015 dump database
  dumpdb_2025.pkl         # Parsed 2025 dump database
  dict_*.pkl              # Dictionaries
  link_graph_*.pkl        # Link graphs
  mention_db_*.pkl        # Mention databases
  models/wiki2015.pkl     # Trained 2015 embeddings
  models/wiki2025.pkl     # Trained 2025 embeddings
  results/                # Experiment output JSONs
  .checkpoints/           # Pipeline checkpoint markers

~/ClaudeExperiment/       # Git repo (pushed to GitHub)
  scripts/                # All pipeline scripts
  skills/                 # GCP pipeline skill
  results/                # Committed results (git tracked)

~/pipeline.log            # Main pipeline log
~/dashboard.py            # Monitoring dashboard server
```

## VM Management

### Stop VM (saves money when not in use)
```bash
gcloud compute instances stop wiki-embeddings-vm --zone=asia-south1-a
```

### Start VM
```bash
gcloud compute instances start wiki-embeddings-vm --zone=asia-south1-a
```

### Resume Pipeline After Restart
```bash
gcloud compute ssh wiki-embeddings-vm --zone=asia-south1-a
# Restart dashboard
nohup python3 ~/dashboard.py > ~/dashboard.log 2>&1 &
# Resume pipeline (checkpoints will skip completed steps)
nohup bash ~/ClaudeExperiment/scripts/run_pipeline.sh > ~/pipeline.log 2>&1 &
```

### Delete VM (when experiment is done)
```bash
gcloud compute instances delete wiki-embeddings-vm --zone=asia-south1-a
gcloud compute firewall-rules delete allow-dashboard
```

## GitHub Repo
- **URL:** https://github.com/sumit7194/ClaudeExperiment (private)
- **Auth:** Personal Access Token configured on VM
- Pipeline auto-pushes results on completion
