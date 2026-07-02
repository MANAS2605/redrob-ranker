# Redrob Hackathon — Candidate Ranking System

A three-stage ML pipeline that ranks 100,000 candidates against a fixed job
description, producing a top-100 shortlist with human-readable reasoning.

## Quick Start (Ranking Only — uses precomputed artifacts)

If you already have the `artifacts/` directory with precomputed parquet files:

```bash
# 1. Install dependencies
pip install pandas numpy pyarrow

# 2. Run ranking (< 15 seconds, no GPU, no network)
python rank.py --input candidates.jsonl --output submission.csv

# 3. Validate
python validate_submission.py submission.csv
```

That's it. `rank.py` loads three precomputed parquet files and produces the
submission CSV. No model inference, no network calls.

---

## Full Pipeline (from scratch on a new machine)

### Prerequisites

- Python 3.10+ (tested on 3.14.3)
- 16 GB RAM
- ~3 GB disk for dependencies + artifacts
- Internet access (only for `pip install` and first model download)

### Step 1: Clone and install

```bash
git clone <your-repo-url>
cd redrob-hackathon

# Create a virtual environment (recommended)
python -m venv .venv

# Activate it:
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies installed:**
- `pandas`, `numpy`, `pyarrow` — data processing
- `sentence-transformers` — local embedding model (brings PyTorch as a dependency)

### Step 2: Place data files

Ensure these files are in the repo root:
- `candidates.jsonl` — the 100K candidate dataset (~487 MB)
- `sample_candidates.json` — the 50-candidate sample (for testing)

### Step 3: Run precomputation (one-time, ~2.5 hours on CPU)

```bash
# Stage 1a: Extract 14 structured features (~40 seconds)
python precompute/precompute_features.py \
    --input candidates.jsonl \
    --output artifacts/features.parquet \
    --summary

# Stage 1b: Run 7 honeypot detection rules (~10 minutes)
python precompute/precompute_honeypot_flags.py \
    --input candidates.jsonl \
    --output artifacts/honeypot_flags.parquet \
    --summary

# Stage 2: Generate semantic embeddings (~2 hours on CPU)
# First run downloads the model (~80 MB) from HuggingFace
python precompute/precompute_embeddings.py \
    --input candidates.jsonl \
    --output artifacts/embeddings.parquet \
    --summary \
    --batch-size 512
```

> **Note:** The embedding step is the slowest (~2 hours on CPU). It uses
> `sentence-transformers/all-MiniLM-L6-v2` which runs entirely locally.
> After the first run, the model is cached in `~/.cache/huggingface/`.

### Step 4: Run ranking (< 15 seconds)

```bash
python rank.py \
    --input candidates.jsonl \
    --output submission.csv \
    --verbose
```

### Step 5: Validate

```bash
python validate_submission.py submission.csv
# Expected output: "Submission is valid."
```

---

## Testing with sample data

To verify the pipeline works before running on the full dataset:

```bash
# Features
python precompute/precompute_features.py \
    --input sample_candidates.json \
    --output artifacts/features.parquet --summary

# Honeypot flags
python precompute/precompute_honeypot_flags.py \
    --input sample_candidates.json \
    --output artifacts/honeypot_flags.parquet --summary

# Embeddings
python precompute/precompute_embeddings.py \
    --input sample_candidates.json \
    --output artifacts/embeddings.parquet --summary

# Rank (will produce < 100 rows since sample has only 50 candidates)
python rank.py --input sample_candidates.json --output submission_sample.csv --verbose
```

**Expected sample results:**
- CAND_0000031 (Recommendation Systems Engineer) should rank #1
- 8 candidates hard-excluded as honeypots
- All 14 unit tests should pass: `python tests/test_honeypot_rules.py`

---

## Running unit tests

```bash
python tests/test_honeypot_rules.py
# Expected: 14 passed, 0 failed
```

---

## Project Structure

```
.
├── rank.py                              # Final ranker (loads parquets, scores, outputs CSV)
├── requirements.txt                     # Python dependencies
├── submission_metadata.yaml             # Hackathon metadata
├── validate_submission.py               # Official submission validator
│
├── precompute/                          # Offline precomputation scripts
│   ├── precompute_features.py           # 14 structured features per candidate
│   ├── precompute_honeypot_flags.py     # 7 plausibility rules
│   └── precompute_embeddings.py         # MiniLM-L6 embeddings + JD similarity
│
├── artifacts/                           # Precomputed data (generated, not checked in)
│   ├── features.parquet                 # 100K rows, 15 columns (~2 MB)
│   ├── honeypot_flags.parquet           # 100K rows (~1.3 MB)
│   └── embeddings.parquet               # 100K rows, 12 columns (~7 MB)
│
├── tests/
│   └── test_honeypot_rules.py           # 14 unit tests for honeypot detection
│
├── candidates.jsonl                     # Input data (not checked in, ~487 MB)
├── sample_candidates.json               # 50-candidate sample for testing
└── sample_submission.csv                # Example submission format
```

---

## Constraint Compliance

| Constraint          | Limit     | Actual                |
|---------------------|-----------|-----------------------|
| Wall-clock runtime  | ≤ 5 min   | **13 seconds**        |
| Memory              | ≤ 16 GB   | **< 500 MB**          |
| Compute             | CPU only  | ✅ No GPU required     |
| Network             | Offline   | ✅ Zero network calls  |
| Disk (intermediate) | ≤ 5 GB    | **~10.5 MB** parquets |

> The 5-minute constraint applies only to `rank.py`. Precomputation
> (`precompute/*.py`) runs once offline with no time limit.

---

## Architecture

```
candidates.jsonl ──┬── precompute_features.py ──────── features.parquet ──┐
                   ├── precompute_honeypot_flags.py ── honeypot.parquet ──┼── rank.py ── submission.csv
                   └── precompute_embeddings.py ─────── embed.parquet ───┘
```

**Scoring formula:** Weighted sum (60% technical fit + 25% profile fit + 15%
logistics) with multiplicative penalties for honeypots, title-chasers, and
consulting-only careers.
