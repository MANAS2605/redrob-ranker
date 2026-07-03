#!/usr/bin/env python3
"""
rank.py — Final Ranking Pipeline

Combines precomputed features, honeypot flags, and embedding similarities
into a final composite score, produces a ranked top-100 shortlist, and
generates the submission CSV with reasoning.

Architecture (designed for 5-minute CPU constraint):
  - Loads 3 precomputed parquet files (features, honeypot_flags, embeddings)
  - Applies weighted scoring rubric to produce final composite score
  - Hard-excludes candidates above honeypot implausibility threshold
  - Ranks remaining candidates by composite score
  - Generates human-readable reasoning per candidate
  - Outputs submission CSV matching validate_submission.py format

Scoring rubric weights (tuned from Stage 1/2 analysis):
  The weights directly map to the prompt's §5 rubric tiers.
  Embedding similarity is the primary signal (catches Tier-5 plain-language
  candidates). Structured features provide guardrails and penalize honeypots.

Usage:
  python rank.py --features artifacts/features.parquet --honeypot artifacts/honeypot_flags.parquet --embeddings artifacts/embeddings.parquet --output submission.csv
  
  Or with all defaults:
  python rank.py --input sample_candidates.json

Constraints satisfied:
  - Wall-clock runtime: <10 seconds (just loading parquets + scoring)
  - Memory: <500MB (parquet files are tiny)
  - CPU only, no network, no GPU
  - Deterministic: tie-breaking by candidate_id ascending
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scoring rubric weights
# ---------------------------------------------------------------------------

# These weights reflect the JD priorities from §5:
# Must-have skills + embeddings are the dominant signals.
# Experience, title, career depth are strong secondary signals.
# Location, notice, behavioral are "fit" signals that differentiate close calls.

WEIGHTS = {
    # Primary signals — technical fit (60% total)
    "sim_combined":           0.25,   # Embedding similarity (catches Tier-5 plain-language)
    "must_have_skill_score":  0.20,   # Must-have skill cluster matches (structured)
    "career_depth_score":     0.10,   # Production ML keyword depth in descriptions
    "nice_to_have_skill_score": 0.05, # Bonus for nice-to-have skills
    
    # Secondary signals — profile fit (25% total)
    "experience_fit_score":   0.08,   # 6-8 years ideal
    "title_relevance_score":  0.07,   # AI/ML title > adjacent tech > non-tech
    "product_vs_services_score": 0.05,# Product companies > IT services
    "education_tier_score":   0.05,   # Tier-1 institution + relevant field
    
    # Tertiary signals — logistics & behavior (15% total)
    "location_fit_score":     0.05,   # Noida/Pune > other India > international
    "notice_period_score":    0.04,   # Short notice preferred
    "behavioral_availability_score": 0.03, # Active on platform, responsive
    "github_score":           0.02,   # GitHub activity (neutral if absent)
    "verification_score":     0.01,   # Email/phone/linkedin verified
}

# Penalties applied multiplicatively (not additive)
TITLE_CHASER_PENALTY_WEIGHT = 0.10     # Max 10% reduction for title-chasers
CONSULTING_ONLY_PENALTY = 0.90          # 10% reduction for all-services career
HONEYPOT_HARD_EXCLUDE_THRESHOLD = 0.30  # Exclude if implausibility >= this


# ---------------------------------------------------------------------------
# Reasoning generator
# ---------------------------------------------------------------------------

# JD-specific skill cluster names for reasoning
MUST_HAVE_CLUSTER_NAMES = [
    "Embeddings/Retrieval",
    "Vector DB/Search Infra",
    "Python/ML Stack",
    "Ranking/Evaluation",
]

NICE_TO_HAVE_CLUSTER_NAMES = [
    "LLM Fine-tuning",
    "Learning to Rank",
    "Infra/Optimization",
    "RAG/LLM Applications",
    "DevOps/Containers",
]


def generate_reasoning(candidate: dict, features: dict, honeypot: dict,
                       embedding: dict, rank: int, score: float,
                       jd_config: dict = None) -> str:
    """
    Generate a human-readable reasoning string for why this candidate
    was ranked at this position.
    
    Per §8 rules:
      - Must reference specific profile facts (years, titles, named skills)
      - Must connect facts to JD requirements
      - No filler, no generic phrases
      - Maximum 500 characters
    """
    parts = []
    profile = candidate["profile"]
    title = profile["current_title"]
    yoe = profile["years_of_experience"]
    location = profile["location"]
    country = profile["country"]
    
    # 1. Core qualification statement
    must_have = features["must_have_skill_score"]
    sim = embedding["sim_combined"]
    
    # Build relevant skill keyword list from JD config (or fallback to AI/ML defaults)
    if jd_config and jd_config.get("must_have_terms"):
        relevant_skill_keywords = [t.lower() for t in jd_config["must_have_terms"]]
    else:
        relevant_skill_keywords = [
            "faiss", "pinecone", "weaviate", "qdrant", "milvus",
            "opensearch", "elasticsearch", "embedding", "vector",
            "pytorch", "tensorflow", "python", "ranking", "retrieval",
            "search", "recommendation", "ndcg", "mrr", "nlp",
            "transformer", "bert", "sentence-transformer",
        ]
    
    # Get experience range description from JD config
    if jd_config and jd_config.get("experience_range"):
        exp_r = jd_config["experience_range"]
        exp_min = exp_r.get("min", 0)
        exp_max = exp_r.get("max", 99)
        if exp_max < 99:
            exp_desc = f"{exp_min}-{exp_max}y"
        elif exp_min > 0:
            exp_desc = f"{exp_min}+y"
        else:
            exp_desc = None
    else:
        exp_desc = "6-8y"
    
    if must_have >= 0.5:
        # Strong skills match — lead with specific skills
        skills = candidate.get("skills", [])
        relevant_skills = []
        for s in skills:
            name_lower = s["name"].lower()
            if any(kw in name_lower for kw in relevant_skill_keywords):
                relevant_skills.append(s["name"])
        
        skill_str = ", ".join(relevant_skills[:5])
        if skill_str:
            parts.append(f"{title} with {yoe:.0f}y exp, strong JD alignment via {skill_str}")
        else:
            parts.append(f"{title} with {yoe:.0f}y exp, strong JD skill alignment")
    elif must_have >= 0.15:
        # Moderate match
        parts.append(f"{title} ({yoe:.0f}y) with partial JD skill overlap")
    else:
        # Weak match — lead with what they do have
        parts.append(f"{title} ({yoe:.0f}y)")
    
    # 2. Experience fit
    exp_fit = features["experience_fit_score"]
    if exp_fit >= 0.85:
        parts.append(f"ideal {yoe:.0f}y experience for this role")
    elif exp_fit < 0.4:
        if yoe < 3:
            if exp_desc:
                parts.append(f"only {yoe:.0f}y experience (JD needs {exp_desc})")
            else:
                parts.append(f"only {yoe:.0f}y experience")
        else:
            if exp_desc:
                parts.append(f"{yoe:.0f}y experience (outside ideal {exp_desc} range)")
            else:
                parts.append(f"{yoe:.0f}y experience")
    
    # 3. Career depth
    depth = features["career_depth_score"]
    if depth >= 0.4:
        parts.append("strong production ML/search depth in career history")
    
    # 4. Title relevance
    title_rel = features["title_relevance_score"]
    if title_rel >= 0.8:
        parts.append(f"directly relevant title: {title}")
    elif title_rel <= 0.1:
        parts.append(f"non-matching title ({title})")
    
    # 5. Location
    loc_fit = features["location_fit_score"]
    if loc_fit >= 1.0:
        parts.append(f"ideal location ({location})")
    elif loc_fit <= 0.2:
        parts.append(f"international ({location}, {country})")
    
    # 6. Notice period
    notice = features["notice_period_score"]
    if notice >= 0.9:
        parts.append("available immediately (<=30d notice)")
    elif notice <= 0.2:
        parts.append("long notice period")
    
    # 7. Product vs services
    prod = features["product_vs_services_score"]
    if features.get("consulting_only_flag"):
        parts.append("all IT-services career (no product co. exp)")
    elif prod >= 0.9:
        parts.append("product company background")
    
    # 8. Honeypot flags
    implausibility = honeypot.get("implausibility_score", 0)
    if implausibility > 0.15:
        reasons = honeypot.get("rule_reasons", "")
        if "Impossible education" in reasons:
            parts.append("education sequence anomaly detected")
    
    # 9. Assessment scores (if any)
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if assessments:
        top_assessed = sorted(assessments.items(), key=lambda x: x[1], reverse=True)[:2]
        assessed_str = ", ".join(f"{k}:{v:.0f}" for k, v in top_assessed)
        parts.append(f"Redrob assessed: {assessed_str}")
    
    # Build final string, truncate to 500 chars
    reasoning = ". ".join(parts)
    if len(reasoning) > 497:
        reasoning = reasoning[:497] + "..."
    
    return reasoning


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------

def compute_composite_score(features: pd.DataFrame, honeypot: pd.DataFrame,
                            embeddings: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all precomputed signals and produce a final composite score.
    
    Returns a DataFrame with candidate_id, composite_score, and all
    component scores for debugging.
    """
    # Merge on candidate_id
    df = features.merge(honeypot[["candidate_id", "implausibility_score", "rule_reasons"]],
                        on="candidate_id", how="left")
    df = df.merge(embeddings[["candidate_id", "sim_combined", "sim_full_max", "sim_skills_max"]],
                  on="candidate_id", how="left")
    
    # Fill missing embedding scores (if candidate wasn't embedded)
    df["sim_combined"] = df["sim_combined"].fillna(0.0)
    df["implausibility_score"] = df["implausibility_score"].fillna(0.0)
    
    # Compute weighted sum
    score = np.zeros(len(df))
    for feature_name, weight in WEIGHTS.items():
        if feature_name in df.columns:
            score += weight * df[feature_name].values
        else:
            print(f"  WARNING: Feature '{feature_name}' not found in data, skipping")
    
    # Apply penalties multiplicatively
    # Title chaser penalty (0 = no penalty, max penalty = TITLE_CHASER_PENALTY_WEIGHT)
    title_chaser = df["title_chaser_penalty"].values
    score *= (1.0 - TITLE_CHASER_PENALTY_WEIGHT * title_chaser)
    
    # Consulting-only penalty
    consulting_mask = df["consulting_only_flag"].values.astype(bool)
    score[consulting_mask] *= CONSULTING_ONLY_PENALTY
    
    df["composite_score"] = score
    
    return df


def rank_and_select(df: pd.DataFrame, candidates: list[dict],
                    top_n: int = 100) -> pd.DataFrame:
    """
    Apply honeypot exclusion, rank by composite score, and select top N.
    
    Tie-breaking: by candidate_id ascending (deterministic, per §2 constraints).
    """
    # Hard-exclude honeypots
    excluded = df["implausibility_score"] >= HONEYPOT_HARD_EXCLUDE_THRESHOLD
    n_excluded = excluded.sum()
    print(f"Hard-excluding {n_excluded} candidates with implausibility >= {HONEYPOT_HARD_EXCLUDE_THRESHOLD}")
    
    eligible = df[~excluded].copy()
    
    # Sort by composite_score DESC, then candidate_id ASC for ties
    eligible = eligible.sort_values(
        ["composite_score", "candidate_id"],
        ascending=[False, True],
    )
    
    # Select top N
    top = eligible.head(top_n).copy()
    top["rank"] = range(1, len(top) + 1)
    
    print(f"Selected top {len(top)} from {len(eligible)} eligible candidates")
    
    return top


def build_submission(top_df: pd.DataFrame, candidates: list[dict],
                     features_df: pd.DataFrame, honeypot_df: pd.DataFrame,
                     embeddings_df: pd.DataFrame,
                     jd_config: dict = None) -> pd.DataFrame:
    """
    Build the submission CSV with columns (in exact order per validator):
      candidate_id, rank, score, reasoning
    """
    # Build candidate lookup
    candidate_lookup = {c["candidate_id"]: c for c in candidates}
    features_lookup = features_df.set_index("candidate_id").to_dict("index")
    honeypot_lookup = honeypot_df.set_index("candidate_id").to_dict("index")
    embeddings_lookup = embeddings_df.set_index("candidate_id").to_dict("index")
    
    rows = []
    for _, row in top_df.iterrows():
        cid = row["candidate_id"]
        rank = row["rank"]
        score_val = row["composite_score"]
        
        candidate = candidate_lookup.get(cid, {})
        feats = features_lookup.get(cid, {})
        honey = honeypot_lookup.get(cid, {})
        emb = embeddings_lookup.get(cid, {})
        
        reasoning = generate_reasoning(candidate, feats, honey, emb, rank, score_val,
                                       jd_config=jd_config)
        
        rows.append({
            "candidate_id": cid,
            "rank": int(rank),
            "score": round(score_val, 4),
            "reasoning": reasoning,
        })
    
    # Enforce column order per validator: candidate_id, rank, score, reasoning
    result = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
    
    # Re-sort after rounding to ensure tie-breaking is correct
    # (rounding can create ties that didn't exist in raw scores)
    result = result.sort_values(
        ["score", "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_candidates(input_path: str) -> list[dict]:
    """Load candidates from JSON array or JSONL file."""
    path = Path(input_path)
    if path.suffix == ".jsonl":
        candidates = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
        return candidates
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates and produce submission CSV"
    )
    parser.add_argument("--input", required=True,
                        help="Path to candidates JSON/JSONL (for reasoning generation)")
    parser.add_argument("--features", default="artifacts/features.parquet",
                        help="Path to precomputed features parquet")
    parser.add_argument("--honeypot", default="artifacts/honeypot_flags.parquet",
                        help="Path to precomputed honeypot flags parquet")
    parser.add_argument("--embeddings", default="artifacts/embeddings.parquet",
                        help="Path to precomputed embeddings/similarity parquet")
    parser.add_argument("--output", default="submission.csv",
                        help="Output submission CSV path")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Number of top candidates to select")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed scoring breakdown")
    parser.add_argument("--jd-config", default=None,
                        help="Path to JD config JSON (from parse_jd.py). "
                             "Used to generate JD-specific reasoning.")
    args = parser.parse_args()
    
    t0 = datetime.now()
    
    # Load precomputed data
    print("Loading precomputed artifacts...")
    features_df = pd.read_parquet(args.features)
    honeypot_df = pd.read_parquet(args.honeypot)
    embeddings_df = pd.read_parquet(args.embeddings)
    print(f"  Features: {len(features_df)} rows")
    print(f"  Honeypot flags: {len(honeypot_df)} rows")
    print(f"  Embeddings: {len(embeddings_df)} rows")
    
    # Load raw candidate data (for reasoning generation)
    print(f"Loading candidates from {args.input}...")
    candidates = parse_candidates(args.input)
    print(f"  Candidates: {len(candidates)}")
    
    # Compute composite scores
    print("\nComputing composite scores...")
    scored_df = compute_composite_score(features_df, honeypot_df, embeddings_df)
    
    # Rank and select
    print("\nRanking and selecting...")
    top_df = rank_and_select(scored_df, candidates, top_n=args.top_n)
    
    # Load JD config if provided (for reasoning)
    jd_config = None
    if args.jd_config:
        print(f"Loading JD config from {args.jd_config}...")
        with open(args.jd_config, "r", encoding="utf-8") as f:
            jd_config = json.load(f)
        print(f"  JD summary: {jd_config.get('jd_summary', 'N/A')}")
    
    # Build submission
    print("\nGenerating submission CSV...")
    submission = build_submission(top_df, candidates, features_df, honeypot_df, embeddings_df,
                                 jd_config=jd_config)
    
    # Save
    submission.to_csv(args.output, index=False)
    
    t1 = datetime.now()
    elapsed = (t1 - t0).total_seconds()
    
    print(f"\nSubmission saved to {args.output}")
    print(f"  Rows: {len(submission)}")
    print(f"  Elapsed: {elapsed:.1f}s")
    
    if args.verbose:
        print("\n" + "=" * 80)
        print("TOP CANDIDATES (detailed)")
        print("=" * 80)
        
        for _, row in submission.iterrows():
            print(f"\n  Rank {row['rank']}: {row['candidate_id']} "
                  f"(score={row['score']:.4f})")
            print(f"    {row['reasoning']}")
    
    # Validation check
    print("\n--- Quick validation ---")
    print(f"  Rows: {len(submission)} (need exactly 100 for full dataset)")
    print(f"  Columns: {list(submission.columns)}")
    print(f"  Unique candidate_ids: {submission['candidate_id'].nunique()}")
    print(f"  Score range: {submission['score'].min():.4f} - "
          f"{submission['score'].max():.4f}")
    max_reasoning_len = submission['reasoning'].str.len().max()
    print(f"  Max reasoning length: {max_reasoning_len} chars (limit: 500)")
    
    if max_reasoning_len > 500:
        print("  WARNING: Some reasoning strings exceed 500 chars!")
    
    if len(submission) != 100:
        print(f"  NOTE: Row count is {len(submission)} (validator needs exactly 100 for full dataset)")


if __name__ == "__main__":
    main()
