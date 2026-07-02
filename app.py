"""
app.py — Streamlit sandbox for the Redrob Candidate Ranking System.

Runs the full pipeline on sample_candidates.json and displays results.
Deployed on HuggingFace Spaces as a Docker container.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Precompute artifacts on first run (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_precomputation():
    """Run the full precomputation pipeline on sample data. Cached so it only runs once."""
    input_file = "sample_candidates.json"
    steps = [
        ("Feature Extraction", [
            sys.executable, "precompute/precompute_features.py",
            "--input", input_file, "--output", "artifacts/features.parquet",
        ]),
        ("Honeypot Detection", [
            sys.executable, "precompute/precompute_honeypot_flags.py",
            "--input", input_file, "--output", "artifacts/honeypot_flags.parquet",
        ]),
        ("Semantic Embeddings", [
            sys.executable, "precompute/precompute_embeddings.py",
            "--input", input_file, "--output", "artifacts/embeddings.parquet",
        ]),
    ]
    
    logs = {}
    for step_name, cmd in steps:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        logs[step_name] = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
        if result.returncode != 0:
            return logs, False
    
    return logs, True


@st.cache_data(show_spinner=False)
def run_ranking():
    """Run rank.py on sample data."""
    cmd = [
        sys.executable, "rank.py",
        "--input", "sample_candidates.json",
        "--output", "submission_demo.csv",
        "--top-n", "50",  # sample has only 50 candidates
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    elapsed = time.time() - t0
    return result, elapsed


@st.cache_data(show_spinner=False)
def load_candidates():
    """Load sample candidates."""
    with open("sample_candidates.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎯 Redrob Candidate Ranker")
st.markdown("""
**AI-powered candidate ranking for the Redrob Hackathon.**  
This sandbox runs the full pipeline on a 50-candidate sample dataset.
""")

st.divider()

# Step 1: Precomputation
with st.status("Running precomputation pipeline...", expanded=True) as status:
    st.write("⚙️ Extracting features, detecting honeypots, computing embeddings...")
    logs, success = run_precomputation()
    
    if success:
        status.update(label="Precomputation complete!", state="complete")
    else:
        status.update(label="Precomputation failed!", state="error")
        for step_name, log in logs.items():
            if log["returncode"] != 0:
                st.error(f"**{step_name}** failed:")
                st.code(log["stderr"])
        st.stop()

# Step 2: Ranking
with st.status("Running ranking pipeline...", expanded=False) as status:
    result, elapsed = run_ranking()
    if result.returncode == 0:
        status.update(label=f"Ranking complete in {elapsed:.1f}s!", state="complete")
    else:
        status.update(label="Ranking failed!", state="error")
        st.code(result.stderr)
        st.stop()

st.divider()

# Results
submission = pd.read_csv("submission_demo.csv")
candidates = load_candidates()
candidate_lookup = {c["candidate_id"]: c for c in candidates}

# Key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Candidates Processed", len(candidates))
with col2:
    honeypot_df = pd.read_parquet("artifacts/honeypot_flags.parquet")
    excluded = (honeypot_df["implausibility_score"] >= 0.3).sum()
    st.metric("Honeypots Excluded", int(excluded))
with col3:
    st.metric("Ranked Candidates", len(submission))
with col4:
    st.metric("Ranking Time", f"{elapsed:.1f}s")

st.divider()

# Ranking results table
st.subheader("📊 Ranked Candidates")

# Build display table
display_rows = []
for _, row in submission.iterrows():
    cand = candidate_lookup.get(row["candidate_id"], {})
    profile = cand.get("profile", {})
    display_rows.append({
        "Rank": int(row["rank"]),
        "Candidate ID": row["candidate_id"],
        "Score": f"{row['score']:.4f}",
        "Title": profile.get("current_title", ""),
        "YOE": f"{profile.get('years_of_experience', 0):.0f}y",
        "Location": profile.get("location", ""),
        "Reasoning": row["reasoning"],
    })

display_df = pd.DataFrame(display_rows)
st.dataframe(display_df, use_container_width=True, hide_index=True)

# Detailed view
st.divider()
st.subheader("🔍 Candidate Details")

selected_rank = st.selectbox(
    "Select a candidate to view details:",
    options=range(1, len(submission) + 1),
    format_func=lambda r: f"Rank {r}: {submission.iloc[r-1]['candidate_id']}",
)

row = submission.iloc[selected_rank - 1]
cand = candidate_lookup.get(row["candidate_id"], {})
profile = cand.get("profile", {})

col_left, col_right = st.columns(2)

with col_left:
    st.markdown(f"### {profile.get('current_title', 'N/A')}")
    st.markdown(f"**Score:** {row['score']:.4f}")
    st.markdown(f"**Experience:** {profile.get('years_of_experience', 0):.1f} years")
    st.markdown(f"**Location:** {profile.get('location', 'N/A')}, {profile.get('country', 'N/A')}")
    st.markdown(f"**Company:** {profile.get('current_company', 'N/A')} ({profile.get('current_industry', 'N/A')})")
    
    st.markdown("**Reasoning:**")
    st.info(row["reasoning"])

with col_right:
    # Skills
    skills = cand.get("skills", [])
    if skills:
        st.markdown("**Skills:**")
        for s in skills:
            dur = s.get("duration_months")
            dur_str = f", {dur}mo" if dur else ""
            st.markdown(f"- {s['name']} ({s.get('proficiency', 'N/A')}{dur_str})")
    
    # Assessments
    assessments = cand.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if assessments:
        st.markdown("**Redrob Assessments:**")
        for skill, score in sorted(assessments.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"- {skill}: **{score:.0f}**/100")

# Pipeline logs (collapsible)
with st.expander("📋 Pipeline Logs"):
    for step_name, log in logs.items():
        st.markdown(f"**{step_name}:**")
        st.code(log["stdout"][-2000:] if len(log["stdout"]) > 2000 else log["stdout"])

st.divider()
st.caption("Built for the Redrob Hackathon. Pipeline: feature extraction → honeypot detection → semantic embeddings → weighted ranking.")
