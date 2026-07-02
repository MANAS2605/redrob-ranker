"""
app.py — Streamlit sandbox for the Redrob Candidate Ranking System.

Supports two modes:
1. Demo mode: runs on bundled sample_candidates.json
2. Upload mode: recruiter uploads their own candidate file (JSON/JSONL)

Deployed on HuggingFace Spaces as a Docker container.
"""

import json
import os
import subprocess
import sys
import tempfile
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
# Pipeline functions
# ---------------------------------------------------------------------------

def run_precomputation(input_file: str):
    """Run the full precomputation pipeline on given input file."""
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


def run_ranking(input_file: str, top_n: int = 100):
    """Run rank.py on given input file."""
    cmd = [
        sys.executable, "rank.py",
        "--input", input_file,
        "--output", "submission_demo.csv",
        "--top-n", str(top_n),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    elapsed = time.time() - t0
    return result, elapsed


def load_candidates(input_file: str):
    """Load candidates from JSON or JSONL file."""
    with open(input_file, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == "[":
            # JSON array
            return json.load(f)
        else:
            # JSONL (one JSON object per line)
            candidates = []
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
            return candidates


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to a temporary location and return the path."""
    upload_path = "uploaded_candidates.json"
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return upload_path


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎯 Redrob Candidate Ranker")
st.markdown("""
**AI-powered candidate ranking system.**  
Upload your own candidate file or try the built-in demo dataset.
""")

st.divider()

# ---------------------------------------------------------------------------
# Data source selection
# ---------------------------------------------------------------------------

col_mode1, col_mode2 = st.columns(2)

with col_mode1:
    st.markdown("### 📤 Upload Candidates")
    uploaded_file = st.file_uploader(
        "Upload a JSON or JSONL candidate file",
        type=["json", "jsonl"],
        help="File should contain candidate objects with profile, skills, education, career_history fields."
    )

with col_mode2:
    st.markdown("### 📦 Use Demo Dataset")
    use_demo = st.button("Run with sample_candidates.json (50 candidates)", use_container_width=True)

# Determine which file to use
input_file = None
run_key = None  # Used to cache results per file

if uploaded_file is not None:
    input_file = save_uploaded_file(uploaded_file)
    run_key = f"upload_{uploaded_file.name}_{uploaded_file.size}"
    st.success(f"Uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.0f} KB)")
elif use_demo or "demo_ran" in st.session_state:
    input_file = "sample_candidates.json"
    run_key = "demo_sample"
    st.session_state["demo_ran"] = True

if input_file is None:
    st.info("👆 Upload a candidate file or click the demo button to get started.")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Count candidates to set top_n
# ---------------------------------------------------------------------------

candidates = load_candidates(input_file)
n_candidates = len(candidates)
top_n = min(100, n_candidates)

st.markdown(f"**Loaded {n_candidates} candidates** from `{os.path.basename(input_file)}`")

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

# Step 1: Precomputation
with st.status("Running precomputation pipeline...", expanded=True) as status:
    st.write("⚙️ Extracting features, detecting honeypots, computing embeddings...")
    st.write(f"Processing {n_candidates} candidates...")
    
    logs, success = run_precomputation(input_file)
    
    if success:
        status.update(label="✅ Precomputation complete!", state="complete")
    else:
        status.update(label="❌ Precomputation failed!", state="error")
        for step_name, log in logs.items():
            if log["returncode"] != 0:
                st.error(f"**{step_name}** failed:")
                st.code(log["stderr"][-3000:])
        st.stop()

# Step 2: Ranking
with st.status("Running ranking pipeline...", expanded=False) as status:
    result, elapsed = run_ranking(input_file, top_n)
    if result.returncode == 0:
        status.update(label=f"✅ Ranking complete in {elapsed:.1f}s!", state="complete")
    else:
        status.update(label="❌ Ranking failed!", state="error")
        st.code(result.stderr[-3000:])
        st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

submission = pd.read_csv("submission_demo.csv")
candidate_lookup = {c["candidate_id"]: c for c in candidates}

# Key metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Candidates Processed", n_candidates)
with col2:
    honeypot_df = pd.read_parquet("artifacts/honeypot_flags.parquet")
    excluded = (honeypot_df["implausibility_score"] >= 0.3).sum()
    st.metric("Honeypots Excluded", int(excluded))
with col3:
    st.metric("Ranked Candidates", len(submission))
with col4:
    st.metric("Ranking Time", f"{elapsed:.1f}s")

st.divider()

# Download button for submission CSV
st.download_button(
    label="📥 Download submission.csv",
    data=open("submission_demo.csv", "r").read(),
    file_name="submission.csv",
    mime="text/csv",
    use_container_width=True,
)

st.divider()

# Ranking results table
st.subheader("📊 Ranked Candidates")

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
        for s in skills[:15]:  # Show top 15 skills
            dur = s.get("duration_months")
            dur_str = f", {dur}mo" if dur else ""
            st.markdown(f"- {s['name']} ({s.get('proficiency', 'N/A')}{dur_str})")
        if len(skills) > 15:
            st.caption(f"... and {len(skills) - 15} more skills")
    
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
        stdout = log["stdout"]
        st.code(stdout[-2000:] if len(stdout) > 2000 else stdout)

st.divider()
st.caption("Built for the Redrob Hackathon • Pipeline: feature extraction → honeypot detection → semantic embeddings → weighted ranking")
