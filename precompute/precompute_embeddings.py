#!/usr/bin/env python3
"""
precompute_embeddings.py — Stage 2: Semantic Embeddings

Generates per-candidate text embeddings and computes similarity scores
against multiple JD persona variants. This defends against the plain-language
Tier-5 trap (§6.2) — a single keyword-heavy JD embedding would systematically
favor keyword-heavy profiles over candidates who describe real production
work in plain English.

Key design decisions:
  1. Text blob construction: headline + summary + skill names weighted heavily.
     Career descriptions are DOWN-WEIGHTED because the dataset has shuffled/
     randomized descriptions (discovered in Stage 1 — titles don't match
     descriptions). We still include them but they're less trusted.
  2. Multi-persona JD matching: 3 persona embeddings of "what a great fit
     looks like" — buzzword-heavy, plain-language production, and hybrid.
     Use max similarity across personas per candidate.
  3. Model: sentence-transformers/all-MiniLM-L6-v2 — fast CPU inference,
     384-dimensional embeddings, ~23M parameters.

Usage:
  python precompute/precompute_embeddings.py --input sample_candidates.json --output artifacts/embeddings.parquet
  python precompute/precompute_embeddings.py --input candidates.jsonl --output artifacts/embeddings.parquet
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# JD Persona definitions — what "great fit" looks like from different angles
# ---------------------------------------------------------------------------

# Persona 1: Buzzword-heavy (catches candidates who use standard terminology)
JD_PERSONA_BUZZWORD = (
    "Senior AI Engineer with deep expertise in embeddings-based retrieval systems, "
    "sentence-transformers, BGE, E5, OpenAI embeddings. Production experience with "
    "vector databases including Pinecone, Weaviate, Qdrant, Milvus, FAISS, "
    "Elasticsearch, OpenSearch. Strong Python, PyTorch, TensorFlow. Designed "
    "evaluation frameworks for ranking systems using NDCG, MRR, MAP metrics. "
    "Experience with A/B testing, offline/online correlation. LLM fine-tuning "
    "with LoRA, QLoRA, PEFT. Learning-to-rank models, LambdaMART. RAG pipelines, "
    "retrieval augmented generation. MLOps, model serving, distributed inference. "
    "HR-tech, talent matching, candidate ranking systems."
)

# Persona 2: Plain-language production engineer (catches Tier-5 candidates
# who describe real work without buzzwords)
JD_PERSONA_PLAIN = (
    "Built and shipped a production search and recommendation system that serves "
    "real users at scale. Designed the end-to-end pipeline from feature extraction "
    "through model training to serving. Owned the evaluation framework — measured "
    "ranking quality with offline metrics, ran A/B tests, correlated offline gains "
    "with online engagement lifts. Worked with text similarity and matching at scale, "
    "handling millions of queries against a large document corpus. Built real-time "
    "and batch processing pipelines. Strong software engineering fundamentals — "
    "writes clean Python, designs systems for reliability and performance, owns "
    "production deployments end to end. 6-8 years of hands-on engineering experience "
    "at product companies, not just consulting."
)

# Persona 3: Hybrid — domain + behavioral (catches the ideal profile described in §5)
JD_PERSONA_HYBRID = (
    "Applied ML engineer at a product company, 6-8 years experience. Shipped an "
    "end-to-end ranking, search, or recommendation system at real scale. Built "
    "vector search infrastructure, designed embedding pipelines, evaluated retrieval "
    "quality. Strong opinions on tradeoffs between dense retrieval, sparse retrieval, "
    "and hybrid approaches. Experience with information retrieval, NLP, transformers. "
    "Comfortable with production systems — monitoring, debugging, scaling. Based in "
    "India, ideally Noida or Pune, open to work and responsive. Short notice period. "
    "Active on platform, good engagement signals. Founded or early team member at "
    "a startup building AI-powered products."
)

JD_PERSONAS = {
    "buzzword": JD_PERSONA_BUZZWORD,
    "plain_language": JD_PERSONA_PLAIN,
    "hybrid": JD_PERSONA_HYBRID,
}


# ---------------------------------------------------------------------------
# Text blob construction
# ---------------------------------------------------------------------------

def build_candidate_text(candidate: dict) -> str:
    """
    Build a rich text representation of a candidate for embedding.
    
    Weighting strategy (via repetition/ordering, since embeddings are
    influenced by token proximity and frequency):
      - Headline: included once (concise, usually informative)
      - Summary: included once (often the richest signal per §3 comment)
      - Skill names: included with proficiency (important for matching)
      - Career descriptions: included but AFTER skills/summary
        (downweighted because descriptions are shuffled in this dataset)
      - Certifications: included briefly
    """
    parts = []
    profile = candidate["profile"]
    
    # 1. Headline — concise professional identity
    headline = profile.get("headline", "").strip()
    if headline:
        parts.append(headline)
    
    # 2. Current title + company context
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    industry = profile.get("current_industry", "")
    yoe = profile.get("years_of_experience", 0)
    if title:
        parts.append(f"{title} with {yoe:.0f} years of experience at {company} in {industry}.")
    
    # 3. Summary — often the richest signal
    summary = profile.get("summary", "").strip()
    if summary:
        parts.append(summary)
    
    # 4. Skills — name + proficiency (important for keyword matching)
    skills = candidate.get("skills", [])
    if skills:
        skill_strs = []
        for s in skills:
            name = s["name"]
            prof = s.get("proficiency", "")
            dur = s.get("duration_months")
            if dur is not None and dur > 0:
                skill_strs.append(f"{name} ({prof}, {dur} months)")
            else:
                skill_strs.append(f"{name} ({prof})")
        parts.append("Skills: " + ", ".join(skill_strs) + ".")
    
    # 5. Redrob skill assessment scores (trusted signal)
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if assessments:
        assessed = [f"{k}: {v:.0f}/100" for k, v in assessments.items()]
        parts.append("Verified skill assessments: " + ", ".join(assessed) + ".")
    
    # 6. Certifications
    certs = candidate.get("certifications", [])
    if certs:
        cert_strs = [f"{c['name']} ({c['issuer']}, {c['year']})" for c in certs]
        parts.append("Certifications: " + ", ".join(cert_strs) + ".")
    
    # 7. Career history descriptions — included but ordered last
    # (less trusted due to shuffled descriptions in dataset)
    for job in candidate.get("career_history", []):
        desc = job.get("description", "").strip()
        if desc:
            job_title = job.get("title", "")
            company_name = job.get("company", "")
            parts.append(f"Role at {company_name} as {job_title}: {desc}")
    
    return " ".join(parts)


def build_candidate_text_skills_only(candidate: dict) -> str:
    """
    A minimal text blob focused on skills and headline only.
    
    This is a secondary embedding to compute a pure skill-match similarity,
    less polluted by career descriptions.
    """
    parts = []
    profile = candidate["profile"]
    
    headline = profile.get("headline", "").strip()
    if headline:
        parts.append(headline)
    
    title = profile.get("current_title", "")
    if title:
        parts.append(title)
    
    skills = candidate.get("skills", [])
    if skills:
        skill_names = [s["name"] for s in skills]
        parts.append("Skills: " + ", ".join(skill_names))
    
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if assessments:
        parts.append("Assessed: " + ", ".join(assessments.keys()))
    
    summary = profile.get("summary", "").strip()
    if summary:
        # Take first 200 chars of summary only
        parts.append(summary[:200])
    
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Embedding + similarity computation
# ---------------------------------------------------------------------------

def load_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load the sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    # Get embedding dimension — API changed across versions
    try:
        dim = model.get_embedding_dimension()
    except (AttributeError, TypeError):
        try:
            dim = model.get_sentence_embedding_dimension()
        except (AttributeError, TypeError):
            dim = "unknown"
    print(f"Model loaded. Embedding dimension: {dim}")
    return model


def compute_cosine_similarity(embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between each row in embeddings and query_embedding.
    
    Both inputs should be L2-normalized (sentence-transformers does this by default).
    """
    # Ensure 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Cosine similarity = dot product when vectors are normalized
    similarities = embeddings @ query_embedding.T
    return similarities.flatten()


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate candidate embeddings and JD similarity scores")
    parser.add_argument("--input", required=True, help="Path to candidates JSON/JSONL file")
    parser.add_argument("--output", default="artifacts/embeddings.parquet",
                        help="Output parquet path for similarity scores")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-transformers model name")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for encoding")
    parser.add_argument("--save-embeddings", action="store_true",
                        help="Also save raw embeddings as .npy file")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary of similarity scores")
    parser.add_argument("--jd-file", default=None,
                        help="Path to a custom JD text file. If provided, persona embeddings "
                             "are derived from this text instead of the hardcoded defaults.")
    args = parser.parse_args()
    
    # Load candidates
    print(f"Loading candidates from {args.input}...")
    candidates = parse_candidates(args.input)
    print(f"Loaded {len(candidates)} candidates.")
    
    # Build text blobs
    print("Building candidate text representations...")
    full_texts = [build_candidate_text(c) for c in candidates]
    skill_texts = [build_candidate_text_skills_only(c) for c in candidates]
    candidate_ids = [c["candidate_id"] for c in candidates]
    
    # Print a sample text for inspection
    print(f"\n--- Sample text blob (first candidate) ---")
    print(full_texts[0][:500] + "..." if len(full_texts[0]) > 500 else full_texts[0])
    print(f"--- (length: {len(full_texts[0])} chars) ---\n")
    
    # Load model
    model = load_model(args.model)
    
    # Encode candidates
    print(f"Encoding {len(candidates)} candidates (full text)...")
    t0 = time.time()
    full_embeddings = model.encode(
        full_texts, 
        batch_size=args.batch_size, 
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    t1 = time.time()
    print(f"Full-text encoding done in {t1 - t0:.1f}s")
    
    print(f"Encoding {len(candidates)} candidates (skills-only)...")
    skill_embeddings = model.encode(
        skill_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    t2 = time.time()
    print(f"Skills-only encoding done in {t2 - t1:.1f}s")
    
    # Encode JD personas — use custom JD file if provided, else hardcoded defaults
    if args.jd_file:
        print(f"Loading custom JD from {args.jd_file}...")
        with open(args.jd_file, "r", encoding="utf-8") as f:
            custom_jd = f.read().strip()
        print(f"  Custom JD length: {len(custom_jd)} chars")
        # Build 3 persona variants from the custom JD text:
        #   1. Direct: the raw JD text (catches exact-match candidates)
        #   2. Rephrased: simplified description of what a great hire looks like
        #   3. Skills focus: extract skill-like keywords and requirements
        custom_personas = {
            "direct": custom_jd,
            "rephrased": (
                f"An experienced engineer who is an excellent fit for this role. "
                f"They have the skills and experience described in the following job description: "
                f"{custom_jd[:800]}"
            ),
            "skills_focus": (
                f"A candidate with strong technical skills matching: {custom_jd[:600]}. "
                f"Production experience, hands-on engineering, and relevant domain expertise."
            ),
        }
        active_personas = custom_personas
    else:
        active_personas = JD_PERSONAS
    
    print(f"Encoding {len(active_personas)} JD persona variants...")
    persona_embeddings = {}
    for name, text in active_personas.items():
        emb = model.encode([text], normalize_embeddings=True)
        persona_embeddings[name] = emb[0]
    
    # Compute similarity scores
    print("Computing similarity scores...")
    results = {"candidate_id": candidate_ids}
    
    # Full-text similarities against each persona
    for persona_name, persona_emb in persona_embeddings.items():
        full_sims = compute_cosine_similarity(full_embeddings, persona_emb)
        skill_sims = compute_cosine_similarity(skill_embeddings, persona_emb)
        
        results[f"sim_full_{persona_name}"] = full_sims
        results[f"sim_skills_{persona_name}"] = skill_sims
    
    # Aggregate similarities
    # Max similarity across personas (full text) — best-case match
    full_sim_cols = [f"sim_full_{name}" for name in active_personas]
    skill_sim_cols = [f"sim_skills_{name}" for name in active_personas]
    
    full_sim_matrix = np.column_stack([results[col] for col in full_sim_cols])
    skill_sim_matrix = np.column_stack([results[col] for col in skill_sim_cols])
    
    results["sim_full_max"] = full_sim_matrix.max(axis=1)
    results["sim_full_mean"] = full_sim_matrix.mean(axis=1)
    results["sim_skills_max"] = skill_sim_matrix.max(axis=1)
    results["sim_skills_mean"] = skill_sim_matrix.mean(axis=1)
    
    # Combined similarity: weighted blend of full-text and skills-only
    # Skills-only gets higher weight because descriptions are shuffled
    results["sim_combined"] = 0.4 * results["sim_full_max"] + 0.6 * results["sim_skills_max"]
    
    # Build DataFrame
    df = pd.DataFrame(results)
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(args.output, index=False)
    print(f"\nSaved similarity scores to {args.output} ({len(df)} rows, {len(df.columns)} columns)")
    
    # Optionally save raw embeddings
    if args.save_embeddings:
        emb_path = args.output.replace(".parquet", "_raw.npz")
        np.savez_compressed(
            emb_path,
            candidate_ids=np.array(candidate_ids),
            full_embeddings=full_embeddings,
            skill_embeddings=skill_embeddings,
        )
        print(f"Saved raw embeddings to {emb_path}")
    
    if args.summary:
        print("\n" + "=" * 80)
        print("SIMILARITY SCORE SUMMARY")
        print("=" * 80)
        
        sim_cols = [c for c in df.columns if c.startswith("sim_")]
        print(df[sim_cols].describe().round(4).to_string())
        
        print("\n--- Top 10 by sim_combined ---")
        top = df.nlargest(10, "sim_combined")
        for _, row in top.iterrows():
            cand = next(c for c in candidates if c["candidate_id"] == row["candidate_id"])
            title = cand["profile"]["current_title"]
            yoe = cand["profile"]["years_of_experience"]
            loc = cand["profile"]["location"]
            print(f"  {row['candidate_id']} ({title}, {yoe:.0f}y, {loc})")
            print(f"    combined={row['sim_combined']:.4f}  "
                  f"full_max={row['sim_full_max']:.4f}  "
                  f"skills_max={row['sim_skills_max']:.4f}")
        
        print("\n--- Bottom 5 by sim_combined ---")
        bottom = df.nsmallest(5, "sim_combined")
        for _, row in bottom.iterrows():
            cand = next(c for c in candidates if c["candidate_id"] == row["candidate_id"])
            title = cand["profile"]["current_title"]
            print(f"  {row['candidate_id']} ({title}): combined={row['sim_combined']:.4f}")
        
        # Check persona coverage — which persona matches best for each candidate?
        print("\n--- Persona that matches best per candidate (full text) ---")
        persona_names = list(active_personas.keys())
        best_persona_idx = full_sim_matrix.argmax(axis=1)
        from collections import Counter
        counts = Counter(persona_names[i] for i in best_persona_idx)
        for persona, count in counts.most_common():
            print(f"  {persona}: {count} candidates matched best by this persona")


if __name__ == "__main__":
    main()
