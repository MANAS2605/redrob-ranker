#!/usr/bin/env python3
"""
precompute_features.py — Stage 1: Feature Engineering

Reads candidate data (JSON array or JSONL) and produces a structured feature
table (parquet) with one row per candidate and normalized 0-1 feature columns.

Features map directly to the JD rubric for Senior AI Engineer (§5 of prompt):
  - must_have_skill_score: match against 4 required skill clusters
  - nice_to_have_skill_score: match against bonus skill clusters
  - experience_fit_score: triangular fit around 6-8 years ideal
  - title_relevance_score: how AI/ML-relevant the candidate's titles are
  - career_depth_score: production ML keywords in career descriptions
  - product_vs_services_score: product company vs IT-services career mix
  - title_chaser_penalty: short tenures + title escalation pattern
  - location_fit_score: proximity to Noida/Pune/other target cities
  - notice_period_score: <=30d preferred, scaled penalty to 180d
  - behavioral_availability_score: recency, response rate, open-to-work
  - github_score: GitHub activity (-1 = neutral, not penalty)
  - verification_score: email/phone/linkedin verified
  - education_tier_score: institution tier + field relevance
  - consulting_only_flag: all IT-services career = hard penalty

Usage:
  python precompute/precompute_features.py --input sample_candidates.json --output artifacts/features.parquet
  python precompute/precompute_features.py --input candidates.jsonl --output artifacts/features.parquet
"""

import argparse
import json
import math
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Skill cluster definitions (case-insensitive matching)
# ---------------------------------------------------------------------------

# Must-have: the JD's 4 pillars
MUST_HAVE_CLUSTERS = {
    "embeddings_retrieval": [
        "sentence-transformers", "sentence transformers", "openai embeddings",
        "bge", "e5", "embeddings", "word2vec", "doc2vec", "text embeddings",
        "embedding", "sbert",
    ],
    "vector_db_search": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
        "elasticsearch", "elastic search", "faiss", "vector database",
        "vector databases", "vector db", "annoy", "chromadb", "chroma",
        "hybrid search", "semantic search",
    ],
    "python_ml_stack": [
        "python", "pytorch", "tensorflow", "numpy", "pandas", "scikit-learn",
        "sklearn", "scipy", "keras", "jax",
    ],
    "ranking_evaluation": [
        "ndcg", "mrr", "map", "ranking", "evaluation framework",
        "a/b testing", "ab testing", "information retrieval",
        "search relevance", "recommendation", "retrieval",
        "learning to rank", "ranking systems", "search systems",
        "recall", "precision", "offline evaluation",
    ],
}

# Nice-to-have skill clusters
NICE_TO_HAVE_CLUSTERS = {
    "llm_finetuning": [
        "lora", "qlora", "peft", "fine-tuning llms", "fine-tuning",
        "llm fine-tuning", "finetuning", "adapter", "instruction tuning",
    ],
    "learning_to_rank": [
        "learning to rank", "lambdamart", "xgboost", "lightgbm",
        "gradient boosting", "catboost",
    ],
    "infra_optimization": [
        "distributed systems", "inference optimization", "mlops",
        "model serving", "triton", "tensorrt", "onnx", "vllm",
        "ray", "dask",
    ],
    "rag_llm": [
        "rag", "retrieval augmented generation", "langchain", "llamaindex",
        "llm", "large language model", "gpt", "chatgpt",
    ],
    "devops": [
        "docker", "kubernetes", "k8s", "ci/cd", "terraform", "helm",
    ],
}

# Proficiency level weights (for skill scoring)
PROFICIENCY_WEIGHT = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.45,
    "beginner": 0.15,
}

# Title relevance tiers (case-insensitive substring matching)
TITLE_TIER_1 = [  # Direct AI/ML engineering roles → 1.0
    "ai engineer", "ml engineer", "machine learning engineer",
    "deep learning engineer", "nlp engineer", "data scientist",
    "research scientist", "research engineer",
    "recommendation systems engineer", "search engineer", "ranking engineer",
    "applied scientist", "ml scientist",
]
TITLE_TIER_2 = [  # Adjacent technical roles → 0.65
    "software engineer", "backend engineer", "data engineer",
    "platform engineer", "infrastructure engineer",
    "full stack developer", "python developer",
]
TITLE_TIER_3 = [  # Somewhat technical → 0.35
    "devops engineer", "cloud engineer", "qa engineer",
    "frontend engineer", "mobile developer", "java developer",
    ".net developer", "web developer", "systems engineer",
]
# Everything else → 0.05

# Production ML keywords for career_depth scoring
PRODUCTION_ML_KEYWORDS = [
    "machine learning", "deep learning", "neural network", "embedding",
    "vector", "recommendation", "ranking", "search", "retrieval",
    "nlp", "natural language", "transformer", "bert", "gpt",
    "model training", "model serving", "inference", "feature engineering",
    "a/b test", "ab test", "pytorch", "tensorflow", "production ml",
    "ml pipeline", "faiss", "elasticsearch", "rag", "llm",
    "fine-tun", "deployed", "production", "scale", "latency",
    "throughput", "real-time", "batch processing", "model evaluation",
    "ndcg", "precision", "recall", "f1", "auc",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "sentence-transformer", "hugging face", "huggingface",
    "knowledge graph", "entity recognition", "text classification",
    "sentiment analysis", "topic model", "word2vec", "doc2vec",
    "attention mechanism", "seq2seq",
]

# IT Services / consulting firm indicators (industry-based, not name-based)
IT_SERVICES_INDUSTRIES = {
    "it services", "it consulting", "consulting", "information technology",
    "it services and it consulting", "information technology & services",
    "staffing and recruiting", "outsourcing",
}

# Location matching (case-insensitive substrings)
IDEAL_LOCATIONS = ["noida", "pune"]
GOOD_LOCATIONS = [
    "hyderabad", "mumbai", "delhi", "new delhi", "gurgaon", "gurugram",
    "bangalore", "bengaluru", "chennai", "kolkata",
]

# Seniority levels for title-chaser detection (rough ordering)
SENIORITY_KEYWORDS = [
    ("intern", 0), ("junior", 1), ("associate", 2),
    ("analyst", 2), ("engineer", 3), ("developer", 3),
    ("senior", 4), ("lead", 5), ("staff", 6),
    ("principal", 7), ("architect", 7), ("director", 8),
    ("vp", 9), ("head", 9), ("cto", 10), ("ceo", 10),
    ("manager", 5),
]


# ---------------------------------------------------------------------------
# Feature extraction functions
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
        # Assume JSON array
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def _match_skill_to_cluster(skill_name_lower: str, cluster_terms: list[str]) -> bool:
    """Check if a skill name matches any term in a cluster (case-insensitive)."""
    for term in cluster_terms:
        # Check both directions: term in skill_name, or skill_name in term
        if term in skill_name_lower or skill_name_lower in term:
            return True
    return False


def compute_must_have_skill_score(candidate: dict) -> float:
    """
    Score 0-1 for the 4 must-have skill clusters.
    
    Each cluster contributes 0.25 max. Within each cluster:
      - Presence of any matching skill: 0.3 of cluster weight
      - Best proficiency level: 0.2 of cluster weight
      - Duration months (if present): 0.2 of cluster weight
      - Redrob assessment score (if present): 0.3 of cluster weight
    """
    skills = candidate.get("skills", [])
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    
    cluster_scores = []
    for cluster_name, cluster_terms in MUST_HAVE_CLUSTERS.items():
        best_proficiency = 0.0
        best_duration_norm = None  # None = no data
        best_assessment = None     # None = no assessment
        has_match = False
        
        for skill in skills:
            skill_name_lower = skill["name"].lower()
            if _match_skill_to_cluster(skill_name_lower, cluster_terms):
                has_match = True
                prof_weight = PROFICIENCY_WEIGHT.get(skill["proficiency"], 0.0)
                best_proficiency = max(best_proficiency, prof_weight)
                
                # Duration months (optional field!)
                dur = skill.get("duration_months")
                if dur is not None:
                    # Normalize: 0-6mo → 0.1, 6-24mo → linear to 0.6, 24-60mo → linear to 1.0
                    if dur <= 6:
                        dur_norm = 0.1
                    elif dur <= 24:
                        dur_norm = 0.1 + 0.5 * (dur - 6) / 18
                    else:
                        dur_norm = min(1.0, 0.6 + 0.4 * (dur - 24) / 36)
                    if best_duration_norm is None or dur_norm > best_duration_norm:
                        best_duration_norm = dur_norm
        
        # Check Redrob assessments for this cluster
        for assessed_skill, score in assessments.items():
            if _match_skill_to_cluster(assessed_skill.lower(), cluster_terms):
                norm_score = score / 100.0
                if best_assessment is None or norm_score > best_assessment:
                    best_assessment = norm_score
                # Assessment presence also counts as a match
                has_match = True
        
        if not has_match:
            cluster_scores.append(0.0)
            continue
        
        # Weighted sub-score within this cluster
        sub = 0.0
        sub += 0.3 * 1.0  # presence
        sub += 0.2 * best_proficiency
        if best_duration_norm is not None:
            sub += 0.2 * best_duration_norm
        else:
            sub += 0.2 * 0.5  # neutral when missing
        if best_assessment is not None:
            sub += 0.3 * best_assessment
        else:
            sub += 0.3 * 0.3  # mild neutral when no assessment
        
        cluster_scores.append(sub)
    
    # Average across 4 clusters, each contributing equally
    return sum(cluster_scores) / len(cluster_scores) if cluster_scores else 0.0


def compute_nice_to_have_skill_score(candidate: dict) -> float:
    """Score 0-1 for nice-to-have skill clusters. Simpler: presence + proficiency."""
    skills = candidate.get("skills", [])
    
    cluster_hits = 0
    total_clusters = len(NICE_TO_HAVE_CLUSTERS)
    
    for cluster_name, cluster_terms in NICE_TO_HAVE_CLUSTERS.items():
        best_prof = 0.0
        for skill in skills:
            skill_name_lower = skill["name"].lower()
            if _match_skill_to_cluster(skill_name_lower, cluster_terms):
                prof = PROFICIENCY_WEIGHT.get(skill["proficiency"], 0.0)
                best_prof = max(best_prof, prof)
        if best_prof > 0:
            cluster_hits += best_prof
    
    return min(1.0, cluster_hits / total_clusters)


def compute_experience_fit_score(years: float) -> float:
    """
    Triangular fit: 1.0 at 6-8 years, decay outside.
    
    - 6-8y: 1.0 (ideal zone)
    - 5-6y or 8-9y: 0.85 (still great)
    - 3-5y or 9-12y: 0.5-0.85 (linear decay)
    - 1-3y or 12-20y: 0.2-0.5 (further decay)
    - <1y or >20y: 0.1 (floor)
    """
    if 6 <= years <= 8:
        return 1.0
    elif 5 <= years < 6:
        return 0.85 + 0.15 * (years - 5)
    elif 8 < years <= 9:
        return 1.0 - 0.15 * (years - 8)
    elif 3 <= years < 5:
        return 0.5 + 0.35 * (years - 3) / 2
    elif 9 < years <= 12:
        return 0.85 - 0.35 * (years - 9) / 3
    elif 1 <= years < 3:
        return 0.2 + 0.3 * (years - 1) / 2
    elif 12 < years <= 20:
        return 0.5 - 0.3 * (years - 12) / 8
    else:
        return 0.1


def _get_title_relevance(title: str) -> float:
    """Map a job title to its relevance tier (0-1)."""
    title_lower = title.lower()
    for keyword in TITLE_TIER_1:
        if keyword in title_lower:
            return 1.0
    for keyword in TITLE_TIER_2:
        if keyword in title_lower:
            return 0.65
    for keyword in TITLE_TIER_3:
        if keyword in title_lower:
            return 0.35
    return 0.05


def compute_title_relevance_score(candidate: dict) -> float:
    """
    Best title relevance across current + historical titles.
    
    Current title weighted 60%, best historical 40%.
    """
    current_score = _get_title_relevance(candidate["profile"]["current_title"])
    
    historical_scores = [
        _get_title_relevance(job["title"])
        for job in candidate.get("career_history", [])
    ]
    best_historical = max(historical_scores) if historical_scores else 0.0
    
    return 0.6 * current_score + 0.4 * best_historical


def compute_career_depth_score(candidate: dict) -> float:
    """
    Production ML keyword density in career descriptions.
    
    Catches plain-language Tier-5 candidates who describe real work
    without buzzwords, as well as genuine ML depth.
    """
    all_text = ""
    for job in candidate.get("career_history", []):
        all_text += " " + job.get("description", "")
    all_text += " " + candidate["profile"].get("summary", "")
    all_text += " " + candidate["profile"].get("headline", "")
    
    all_text_lower = all_text.lower()
    
    if len(all_text_lower) < 10:
        return 0.0
    
    hit_count = 0
    matched_keywords = set()
    for kw in PRODUCTION_ML_KEYWORDS:
        if kw in all_text_lower and kw not in matched_keywords:
            hit_count += 1
            matched_keywords.add(kw)
    
    # Normalize: 0 hits → 0.0, 5 hits → 0.3, 10 → 0.6, 15+ → ~0.9, 20+ → 1.0
    # Using a log-like curve to avoid over-rewarding keyword stuffing
    if hit_count == 0:
        return 0.0
    return min(1.0, 0.15 * math.log2(hit_count + 1))


def compute_product_vs_services_score(candidate: dict) -> float:
    """
    1.0 = all product companies, 0.0 = all IT services.
    
    Based on career_history[].industry field, not company names.
    """
    history = candidate.get("career_history", [])
    if not history:
        return 0.5  # no data → neutral
    
    product_months = 0
    services_months = 0
    
    for job in history:
        industry_lower = job.get("industry", "").lower()
        duration = job.get("duration_months", 0)
        
        if industry_lower in IT_SERVICES_INDUSTRIES:
            services_months += duration
        else:
            product_months += duration
    
    total = product_months + services_months
    if total == 0:
        return 0.5
    
    return product_months / total


def compute_title_chaser_penalty(candidate: dict) -> float:
    """
    Detect title-chaser pattern: short tenures + rising seniority.
    
    Returns 0.0 (no penalty) to 1.0 (worst offender).
    """
    history = candidate.get("career_history", [])
    if len(history) < 2:
        return 0.0  # Can't be a title-chaser with 1 role
    
    # Average tenure
    durations = [job["duration_months"] for job in history]
    avg_tenure = sum(durations) / len(durations)
    
    # Check for seniority escalation
    def _estimate_seniority(title: str) -> int:
        title_lower = title.lower()
        best = 3  # default: mid-level
        for keyword, level in SENIORITY_KEYWORDS:
            if keyword in title_lower:
                best = max(best, level)
        return best
    
    # Sort by start_date to check trajectory
    sorted_history = sorted(history, key=lambda j: j["start_date"])
    seniority_levels = [_estimate_seniority(j["title"]) for j in sorted_history]
    
    # Count upward moves
    upward_moves = sum(
        1 for i in range(len(seniority_levels) - 1)
        if seniority_levels[i + 1] > seniority_levels[i]
    )
    
    # Penalty: short tenure + many upward moves
    if avg_tenure >= 24:  # >=2 years average is fine
        return 0.0
    
    # 18mo average + escalation = mild flag
    # 12mo average + escalation = strong flag
    tenure_factor = max(0.0, 1.0 - avg_tenure / 24.0)  # 0 at 24mo, 1 at 0mo
    escalation_factor = min(1.0, upward_moves / max(1, len(history) - 1))
    
    return tenure_factor * escalation_factor


def compute_location_fit_score(candidate: dict) -> float:
    """
    Location fit tiers:
      Noida/Pune = 1.0
      Hyderabad/Mumbai/Delhi NCR/Bangalore = 0.85
      Other India = 0.6
      International + willing_to_relocate = 0.4
      International + no relocate = 0.15
    """
    location = candidate["profile"].get("location", "").lower()
    country = candidate["profile"].get("country", "").lower()
    willing = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)
    
    for city in IDEAL_LOCATIONS:
        if city in location:
            return 1.0
    
    for city in GOOD_LOCATIONS:
        if city in location:
            return 0.85
    
    if country == "india":
        return 0.6
    
    # International
    if willing:
        return 0.4
    return 0.15


def compute_notice_period_score(candidate: dict) -> float:
    """
    Notice period fit:
      <=30 days: 1.0
      30-90 days: linear decay to 0.3
      90-180 days: linear decay to 0.1
    """
    notice = candidate.get("redrob_signals", {}).get("notice_period_days", 90)
    
    if notice <= 30:
        return 1.0
    elif notice <= 90:
        return 1.0 - 0.7 * (notice - 30) / 60
    else:
        return 0.3 - 0.2 * min(1.0, (notice - 90) / 90)


def compute_behavioral_availability_score(candidate: dict) -> float:
    """
    Composite behavioral score from redrob_signals.
    
    Components (each 0-1, averaged with missing-value handling):
      - Recency: days since last_active_date
      - recruiter_response_rate: direct 0-1
      - open_to_work_flag: bool → 1.0/0.0
      - interview_completion_rate: direct 0-1
      - offer_acceptance_rate: -1 = exclude from average
    
    Sentinel values (-1) are excluded from the average, not penalized.
    """
    signals = candidate.get("redrob_signals", {})
    
    components = []
    weights = []
    
    # Recency of last_active_date
    last_active = signals.get("last_active_date")
    if last_active:
        try:
            last_dt = datetime.strptime(last_active, "%Y-%m-%d").date()
            days_ago = (date.today() - last_dt).days
            # 0 days ago = 1.0, 30 days = 0.8, 90 days = 0.5, 365+ days = 0.1
            if days_ago <= 0:
                recency = 1.0
            elif days_ago <= 30:
                recency = 1.0 - 0.2 * days_ago / 30
            elif days_ago <= 90:
                recency = 0.8 - 0.3 * (days_ago - 30) / 60
            elif days_ago <= 365:
                recency = 0.5 - 0.4 * (days_ago - 90) / 275
            else:
                recency = 0.1
            components.append(recency)
            weights.append(0.3)
        except (ValueError, TypeError):
            pass
    
    # Recruiter response rate
    rr = signals.get("recruiter_response_rate")
    if rr is not None and rr >= 0:
        components.append(rr)
        weights.append(0.3)
    
    # Open to work
    otw = signals.get("open_to_work_flag")
    if otw is not None:
        components.append(1.0 if otw else 0.3)
        weights.append(0.15)
    
    # Interview completion rate
    icr = signals.get("interview_completion_rate")
    if icr is not None and icr >= 0:
        components.append(icr)
        weights.append(0.15)
    
    # Offer acceptance rate — sentinel -1 = exclude
    oar = signals.get("offer_acceptance_rate")
    if oar is not None and oar >= 0:
        components.append(oar)
        weights.append(0.1)
    
    if not components:
        return 0.5  # No signal → neutral
    
    # Weighted average
    total_weight = sum(weights)
    return sum(c * w for c, w in zip(components, weights)) / total_weight


def compute_github_score(candidate: dict) -> float:
    """
    GitHub activity: -1 = neutral (0.5), 0-100 → 0.0-1.0.
    
    Never penalizes candidates without GitHub.
    """
    score = candidate.get("redrob_signals", {}).get("github_activity_score", -1)
    if score < 0:
        return 0.5  # No signal, not a penalty
    return score / 100.0


def compute_verification_score(candidate: dict) -> float:
    """Average of email/phone/linkedin verification."""
    signals = candidate.get("redrob_signals", {})
    checks = [
        signals.get("verified_email", False),
        signals.get("verified_phone", False),
        signals.get("linkedin_connected", False),
    ]
    return sum(1.0 for c in checks if c) / 3.0


def compute_education_tier_score(candidate: dict) -> float:
    """
    Best education tier + field-of-study relevance bonus.
    
    Tiers: tier_1=1.0, tier_2=0.75, tier_3=0.5, tier_4=0.3, unknown/missing=0.4
    CS/AI/ML/Data Science field bonus: +0.1
    """
    education = candidate.get("education", [])
    if not education:
        return 0.3  # No education data
    
    TIER_MAP = {
        "tier_1": 1.0,
        "tier_2": 0.75,
        "tier_3": 0.5,
        "tier_4": 0.3,
        "unknown": 0.4,
    }
    
    RELEVANT_FIELDS = {
        "computer science", "artificial intelligence", "machine learning",
        "data science", "information technology", "electronics",
        "electrical engineering", "mathematics", "statistics",
        "computational", "software engineering",
    }
    
    best_score = 0.0
    for edu in education:
        tier = edu.get("tier", "unknown")
        base = TIER_MAP.get(tier, 0.4)
        
        # Field relevance bonus
        field = edu.get("field_of_study", "").lower()
        field_bonus = 0.1 if any(f in field for f in RELEVANT_FIELDS) else 0.0
        
        best_score = max(best_score, base + field_bonus)
    
    return min(1.0, best_score)


def compute_consulting_only_flag(candidate: dict) -> bool:
    """
    True if ALL career history entries are at IT-services/consulting firms.
    
    Based on industry field, not company names (more generalizable).
    One product-company entry anywhere in history clears the flag.
    """
    history = candidate.get("career_history", [])
    if not history:
        return False
    
    for job in history:
        industry_lower = job.get("industry", "").lower()
        if industry_lower not in IT_SERVICES_INDUSTRIES:
            return False  # Has at least one non-services role
    
    return True


def extract_features(candidate: dict) -> dict:
    """Extract all features for a single candidate, returning a flat dict."""
    cid = candidate["candidate_id"]
    
    return {
        "candidate_id": cid,
        "must_have_skill_score": compute_must_have_skill_score(candidate),
        "nice_to_have_skill_score": compute_nice_to_have_skill_score(candidate),
        "experience_fit_score": compute_experience_fit_score(
            candidate["profile"]["years_of_experience"]
        ),
        "title_relevance_score": compute_title_relevance_score(candidate),
        "career_depth_score": compute_career_depth_score(candidate),
        "product_vs_services_score": compute_product_vs_services_score(candidate),
        "title_chaser_penalty": compute_title_chaser_penalty(candidate),
        "location_fit_score": compute_location_fit_score(candidate),
        "notice_period_score": compute_notice_period_score(candidate),
        "behavioral_availability_score": compute_behavioral_availability_score(candidate),
        "github_score": compute_github_score(candidate),
        "verification_score": compute_verification_score(candidate),
        "education_tier_score": compute_education_tier_score(candidate),
        "consulting_only_flag": compute_consulting_only_flag(candidate),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract features from candidate data")
    parser.add_argument("--input", required=True, help="Path to candidates JSON/JSONL file")
    parser.add_argument("--output", default="artifacts/features.parquet",
                        help="Output parquet path")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary statistics after extraction")
    args = parser.parse_args()
    
    print(f"Loading candidates from {args.input}...")
    candidates = parse_candidates(args.input)
    print(f"Loaded {len(candidates)} candidates.")
    
    print("Extracting features...")
    rows = [extract_features(c) for c in candidates]
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(args.output, index=False)
    print(f"Saved features to {args.output} ({len(df)} rows, {len(df.columns)} columns)")
    
    if args.summary:
        print("\n" + "=" * 80)
        print("FEATURE SUMMARY")
        print("=" * 80)
        
        # Numeric columns only
        numeric_cols = [c for c in df.columns if c != "candidate_id" and c != "consulting_only_flag"]
        print(df[numeric_cols].describe().round(3).to_string())
        
        print(f"\nConsulting-only candidates: {df['consulting_only_flag'].sum()}/{len(df)}")
        
        # Top candidates by must-have skill score
        print("\n--- Top 10 by must_have_skill_score ---")
        top = df.nlargest(10, "must_have_skill_score")
        for _, row in top.iterrows():
            print(f"  {row['candidate_id']}: must_have={row['must_have_skill_score']:.3f}, "
                  f"title={row['title_relevance_score']:.3f}, "
                  f"depth={row['career_depth_score']:.3f}, "
                  f"exp={row['experience_fit_score']:.3f}")
        
        # Bottom 5 by must-have
        print("\n--- Bottom 5 by must_have_skill_score ---")
        bottom = df.nsmallest(5, "must_have_skill_score")
        for _, row in bottom.iterrows():
            print(f"  {row['candidate_id']}: must_have={row['must_have_skill_score']:.3f}, "
                  f"title={row['title_relevance_score']:.3f}")


if __name__ == "__main__":
    main()
