#!/usr/bin/env python3
"""
precompute_honeypot_flags.py — Stage 1: Honeypot / Anomaly Detection

Produces an implausibility_score (0-1) per candidate based on rule-based
plausibility checks. Candidates scoring >0.5 are likely honeypots and should
be hard-excluded from the top 100.

Rules and their reasoning (all documented for interview defense):

1. YOE_MISMATCH: years_of_experience vs sum of career_history durations
   - Threshold: >3 years mismatch (allows gaps, sabbaticals)
   - Weight: 0.25

2. EXPERT_SKILL_LOW_DURATION: expert/advanced proficiency with <6 months
   - Only when duration_months is PRESENT (optional field per schema)
   - Weight: 0.20 per flagged skill, capped

3. IMPOSSIBLE_EDU_SEQUENCE: degree ordering violations
   - Ph.D before B.Tech, postgrad end_year < undergrad end_year
   - Weight: 0.30 (strong signal)

4. EDU_CAREER_ANOMALY: undergrad completing years after career start
   - Only for undergraduate degrees, not postgrad (working while studying is normal)
   - Weight: 0.15

5. MULTIPLE_CURRENT_ROLES: >1 career_history entry with is_current=true
   - Weight: 0.10

6. CAREER_DATE_OVERLAP: overlapping full-time roles with >6mo overlap
   - Weight: 0.10

7. TITLE_DESCRIPTION_MISMATCH: title category doesn't match description content
   - Weight: 0.15

Usage:
  python precompute/precompute_honeypot_flags.py --input sample_candidates.json --output artifacts/honeypot_flags.parquet
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Degree classification for education sequence checks
# ---------------------------------------------------------------------------

# Rough ordering: higher number = more advanced degree
DEGREE_LEVEL = {
    # Undergraduate
    "b.tech": 1, "b.e.": 1, "b.sc": 1, "b.a.": 1, "b.com": 1, "bba": 1,
    "bachelor": 1, "bachelors": 1, "b.s.": 1, "b.eng": 1,
    # Postgraduate
    "m.tech": 2, "m.e.": 2, "m.sc": 2, "m.s.": 2, "m.a.": 2, "mba": 2,
    "m.com": 2, "master": 2, "masters": 2, "m.eng": 2, "pgdm": 2,
    # Doctoral
    "ph.d": 3, "phd": 3, "doctorate": 3, "d.phil": 3,
}

# Technical keywords in career descriptions (for title-description mismatch)
TECH_KEYWORDS = {
    "code", "programming", "software", "engineering", "api", "database",
    "algorithm", "deploy", "system design", "architecture", "debug",
    "testing", "unit test", "integration", "pipeline", "framework",
    "python", "java", "javascript", "react", "node", "sql", "nosql",
    "cloud", "aws", "gcp", "azure", "docker", "kubernetes",
    "machine learning", "deep learning", "model", "training", "inference",
    "data pipeline", "etl", "analytics", "data engineering",
}

NON_TECH_KEYWORDS = {
    "hr", "human resources", "recruitment", "hiring", "talent",
    "accounting", "financial reporting", "audit", "tax", "ledger",
    "marketing", "campaign", "brand", "content writing", "seo",
    "operations", "logistics", "warehouse", "fulfillment", "supply chain",
    "customer support", "ticket", "escalation", "support agent",
    "sales", "revenue", "quota", "crm", "cold call",
    "graphic design", "photoshop", "illustrator", "figma", "branding",
    "mechanical", "cad", "solidworks", "ansys", "manufacturing",
    "civil", "structural", "construction", "surveying",
}

# Title categories
TECH_TITLES = {
    "engineer", "developer", "programmer", "architect", "scientist",
    "analyst", "data", "devops", "cloud", "qa", "sre",
}
NON_TECH_TITLES = {
    "hr manager", "accountant", "marketing manager", "operations manager",
    "graphic designer", "civil engineer", "mechanical engineer",
    "customer support", "sales executive", "content writer", "project manager",
}


# ---------------------------------------------------------------------------
# Honeypot detection rules
# ---------------------------------------------------------------------------

def check_yoe_mismatch(candidate: dict) -> tuple[float, str]:
    """
    Rule 1: years_of_experience vs sum of career_history durations.
    
    career_history[].duration_months is always present per schema.
    Threshold: >3 years mismatch to allow for gaps/sabbaticals.
    """
    yoe_claimed = candidate["profile"]["years_of_experience"]
    career_months = sum(
        job["duration_months"] for job in candidate.get("career_history", [])
    )
    yoe_from_career = career_months / 12.0
    
    mismatch = abs(yoe_claimed - yoe_from_career)
    
    if mismatch > 5:
        return 0.25, f"YOE mismatch: claimed {yoe_claimed:.1f}y but career sums to {yoe_from_career:.1f}y (diff={mismatch:.1f}y)"
    elif mismatch > 3:
        # Partial score for moderate mismatch
        return 0.15, f"YOE mismatch (moderate): claimed {yoe_claimed:.1f}y vs career {yoe_from_career:.1f}y (diff={mismatch:.1f}y)"
    
    return 0.0, ""


def check_expert_skill_low_duration(candidate: dict) -> tuple[float, str]:
    """
    Rule 2: expert/advanced proficiency with very low duration_months.
    
    CRITICAL: Only fires when duration_months IS PRESENT on the skill entry.
    Missing duration_months = no signal, NOT 0.
    """
    flags = []
    for skill in candidate.get("skills", []):
        proficiency = skill.get("proficiency", "")
        duration = skill.get("duration_months")  # Can be None/missing!
        
        if duration is None:
            continue  # No signal — do NOT treat as 0
        
        if proficiency == "expert" and duration <= 3:
            flags.append(f"'{skill['name']}' claimed expert with only {duration}mo")
        elif proficiency == "advanced" and duration <= 2:
            flags.append(f"'{skill['name']}' claimed advanced with only {duration}mo")
    
    if not flags:
        return 0.0, ""
    
    # Cap contribution: 0.10 per flagged skill, max 0.25
    score = min(0.25, 0.10 * len(flags))
    return score, "Expert/advanced skills with implausibly low duration: " + "; ".join(flags)


def _classify_degree_level(degree: str) -> int:
    """Return degree level (1=UG, 2=PG, 3=PhD) or 0 if unknown."""
    degree_lower = degree.lower().strip().rstrip(".")
    # Add trailing period variants
    for key, level in DEGREE_LEVEL.items():
        if key in degree_lower or degree_lower in key:
            return level
    return 0


def check_impossible_edu_sequence(candidate: dict) -> tuple[float, str]:
    """
    Rule 3: Impossible degree ordering.
    
    E.g., Ph.D completed before B.Tech, or M.Sc end_year < B.Tech end_year
    when the degree levels should be sequential.
    
    This is a strong honeypot signal.
    """
    education = candidate.get("education", [])
    if len(education) < 2:
        return 0.0, ""
    
    # Build list of (degree_level, end_year, degree_name)
    edu_with_levels = []
    for edu in education:
        level = _classify_degree_level(edu.get("degree", ""))
        end_year = edu.get("end_year", 0)
        if level > 0 and end_year > 0:
            edu_with_levels.append((level, end_year, edu.get("degree", "")))
    
    if len(edu_with_levels) < 2:
        return 0.0, ""
    
    # Sort by degree level (should also be sorted by end_year for valid profiles)
    edu_with_levels.sort(key=lambda x: x[0])
    
    flags = []
    for i in range(len(edu_with_levels) - 1):
        lower_level, lower_year, lower_deg = edu_with_levels[i]
        higher_level, higher_year, higher_deg = edu_with_levels[i + 1]
        
        if lower_level < higher_level and lower_year > higher_year:
            # Lower degree completed AFTER higher degree — impossible
            flags.append(
                f"{lower_deg} (level {lower_level}, graduated {lower_year}) completed "
                f"after {higher_deg} (level {higher_level}, graduated {higher_year})"
            )
    
    if not flags:
        return 0.0, ""
    
    return 0.30, "Impossible education sequence: " + "; ".join(flags)


def check_edu_career_anomaly(candidate: dict) -> tuple[float, str]:
    """
    Rule 4: Undergraduate degree completing years after career start.
    
    Working while doing a postgrad is normal and NOT flagged.
    An undergrad completing 5+ years after career start is suspicious.
    """
    education = candidate.get("education", [])
    career_history = candidate.get("career_history", [])
    
    if not education or not career_history:
        return 0.0, ""
    
    # Find earliest career start
    earliest_start = min(job["start_date"] for job in career_history)
    career_start_year = int(earliest_start.split("-")[0])
    
    flags = []
    for edu in education:
        level = _classify_degree_level(edu.get("degree", ""))
        end_year = edu.get("end_year", 0)
        
        # Only flag undergraduate degrees (level 1) that complete way after career start
        if level == 1 and end_year > 0:
            gap = end_year - career_start_year
            if gap > 3:  # UG completing >3 years after career started
                flags.append(
                    f"UG degree ({edu.get('degree', '')}) graduated {end_year}, "
                    f"but career started {career_start_year} (gap={gap}y)"
                )
    
    if not flags:
        return 0.0, ""
    
    return 0.15, "Education-career timeline anomaly: " + "; ".join(flags)


def check_multiple_current_roles(candidate: dict) -> tuple[float, str]:
    """Rule 5: More than one career_history entry with is_current=true."""
    current_count = sum(
        1 for job in candidate.get("career_history", [])
        if job.get("is_current", False)
    )
    
    if current_count > 1:
        return 0.10, f"Multiple current roles: {current_count} entries marked is_current=true"
    return 0.0, ""


def check_career_date_overlap(candidate: dict) -> tuple[float, str]:
    """
    Rule 6: Overlapping full-time roles with significant overlap.
    
    Small overlaps (transition periods <3 months) are normal and ignored.
    """
    history = candidate.get("career_history", [])
    if len(history) < 2:
        return 0.0, ""
    
    # Parse dates and sort
    dated_jobs = []
    for job in history:
        try:
            start = datetime.strptime(job["start_date"], "%Y-%m-%d")
            end_str = job.get("end_date")
            end = datetime.strptime(end_str, "%Y-%m-%d") if end_str else datetime.now()
            dated_jobs.append((start, end, job["company"], job["title"]))
        except (ValueError, TypeError):
            continue
    
    dated_jobs.sort(key=lambda x: x[0])
    
    flags = []
    for i in range(len(dated_jobs)):
        for j in range(i + 1, len(dated_jobs)):
            s1, e1, c1, t1 = dated_jobs[i]
            s2, e2, c2, t2 = dated_jobs[j]
            
            # Check for overlap
            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)
            
            if overlap_start < overlap_end:
                overlap_days = (overlap_end - overlap_start).days
                if overlap_days > 180:  # >6 months overlap = suspicious
                    flags.append(
                        f"{c1}/{t1} and {c2}/{t2} overlap by {overlap_days} days"
                    )
    
    if not flags:
        return 0.0, ""
    
    return min(0.15, 0.10 * len(flags)), "Career date overlap: " + "; ".join(flags)


def check_title_description_mismatch(candidate: dict) -> tuple[float, str]:
    """
    Rule 7: Title category doesn't match description content.
    
    E.g., title says "HR Manager" but description talks about "built data pipelines."
    Or title says "Software Engineer" but description is about "audit and tax filings."
    
    This catches keyword-stuffing or synthetic/shuffled profiles.
    """
    flags = []
    
    for job in candidate.get("career_history", []):
        title_lower = job["title"].lower()
        desc_lower = job.get("description", "").lower()
        
        if len(desc_lower) < 20:
            continue
        
        # Count tech vs non-tech keywords in description
        tech_count = sum(1 for kw in TECH_KEYWORDS if kw in desc_lower)
        non_tech_count = sum(1 for kw in NON_TECH_KEYWORDS if kw in desc_lower)
        
        # Check if title is tech but description is non-tech
        is_tech_title = any(t in title_lower for t in TECH_TITLES)
        is_non_tech_title = any(t in title_lower for t in NON_TECH_TITLES)
        
        if is_tech_title and non_tech_count > tech_count and non_tech_count >= 3:
            flags.append(
                f"'{job['title']}' at {job['company']}: title is technical but "
                f"description is non-technical ({non_tech_count} non-tech vs {tech_count} tech keywords)"
            )
        elif is_non_tech_title and tech_count > non_tech_count and tech_count >= 3:
            # This is LESS suspicious — non-tech titles can have tech work
            # Only flag if it's extreme
            if tech_count >= 5 and non_tech_count == 0:
                flags.append(
                    f"'{job['title']}' at {job['company']}: title is non-technical but "
                    f"description is purely technical ({tech_count} tech keywords)"
                )
    
    if not flags:
        return 0.0, ""
    
    return min(0.08, 0.04 * len(flags)), "Title-description mismatch: " + "; ".join(flags)


# ---------------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------------

ALL_RULES = [
    ("yoe_mismatch", check_yoe_mismatch),
    ("expert_skill_low_duration", check_expert_skill_low_duration),
    ("impossible_edu_sequence", check_impossible_edu_sequence),
    ("edu_career_anomaly", check_edu_career_anomaly),
    ("multiple_current_roles", check_multiple_current_roles),
    ("career_date_overlap", check_career_date_overlap),
    ("title_description_mismatch", check_title_description_mismatch),
]


def detect_honeypot(candidate: dict) -> dict:
    """
    Run all honeypot detection rules on a candidate.
    
    Returns a dict with:
      - candidate_id
      - implausibility_score (0-1, sum of rule scores, capped at 1.0)
      - One boolean column per rule (True if that rule fired)
      - rule_reasons: list of human-readable strings explaining each flag
    """
    cid = candidate["candidate_id"]
    result = {"candidate_id": cid}
    
    total_score = 0.0
    reasons = []
    
    for rule_name, rule_fn in ALL_RULES:
        score, reason = rule_fn(candidate)
        result[f"flag_{rule_name}"] = score > 0
        total_score += score
        if reason:
            reasons.append(reason)
    
    result["implausibility_score"] = min(1.0, total_score)
    result["rule_reasons"] = "; ".join(reasons) if reasons else ""
    
    return result


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
    parser = argparse.ArgumentParser(description="Detect honeypot/anomalous candidates")
    parser.add_argument("--input", required=True, help="Path to candidates JSON/JSONL file")
    parser.add_argument("--output", default="artifacts/honeypot_flags.parquet",
                        help="Output parquet path")
    parser.add_argument("--summary", action="store_true",
                        help="Print detailed summary of flagged candidates")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Implausibility threshold for reporting (default 0.3)")
    args = parser.parse_args()
    
    print(f"Loading candidates from {args.input}...")
    candidates = parse_candidates(args.input)
    print(f"Loaded {len(candidates)} candidates.")
    
    print("Running honeypot detection...")
    rows = [detect_honeypot(c) for c in candidates]
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(args.output, index=False)
    print(f"Saved honeypot flags to {args.output}")
    
    # Summary stats
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    flagged_any = df[flag_cols].any(axis=1).sum()
    above_threshold = (df["implausibility_score"] >= args.threshold).sum()
    
    print(f"\nFlagged by at least one rule: {flagged_any}/{len(df)}")
    print(f"Above threshold ({args.threshold}): {above_threshold}/{len(df)}")
    
    # Per-rule stats
    print("\nPer-rule flag counts:")
    for col in flag_cols:
        count = df[col].sum()
        rule_name = col.replace("flag_", "")
        print(f"  {rule_name}: {count}")
    
    if args.summary:
        print("\n" + "=" * 80)
        print("FLAGGED CANDIDATES (detailed)")
        print("=" * 80)
        
        flagged = df[df["implausibility_score"] > 0].sort_values(
            "implausibility_score", ascending=False
        )
        
        for _, row in flagged.iterrows():
            cid = row["candidate_id"]
            # Find original candidate data for context
            cand = next(c for c in candidates if c["candidate_id"] == cid)
            title = cand["profile"]["current_title"]
            yoe = cand["profile"]["years_of_experience"]
            
            print(f"\n{cid} ({title}, {yoe}y) -- score: {row['implausibility_score']:.2f}")
            
            # Show which rules fired
            for col in flag_cols:
                if row[col]:
                    print(f"  [X] {col.replace('flag_', '')}")
            
            # Show reasons
            if row["rule_reasons"]:
                for reason in row["rule_reasons"].split("; "):
                    print(f"    -> {reason}")
            
            # Classification
            if row["implausibility_score"] >= 0.5:
                print(f"  ** LIKELY HONEYPOT -- recommend hard exclude **")
            elif row["implausibility_score"] >= args.threshold:
                print(f"  * SUSPICIOUS -- recommend heavy penalty *")
            else:
                print(f"  (i) MILD FLAG -- minor penalty or manual review")


if __name__ == "__main__":
    main()
