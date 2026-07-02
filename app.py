"""
TalentRank AI — Premium AI-Powered Candidate Ranking Interface

A modern, minimal SaaS-style web application that helps recruiters
identify the best candidates using semantic understanding.

Deployed on HuggingFace Spaces via Docker.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TalentRank AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — Premium SaaS styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Global */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hero section */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #EEF2FF, #E0E7FF);
        color: #4F46E5;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #64748B;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto 2rem;
        line-height: 1.6;
    }
    
    /* Step indicators */
    .steps-container {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin: 1.5rem 0 2rem;
        flex-wrap: wrap;
    }
    
    .step-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 20px;
        border-radius: 24px;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .step-active {
        background: linear-gradient(135deg, #4F46E5, #6366F1);
        color: white;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
    }
    
    .step-complete {
        background: #10B981;
        color: white;
    }
    
    .step-inactive {
        background: #E2E8F0;
        color: #94A3B8;
    }
    
    .step-number {
        width: 22px;
        height: 22px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    .step-active .step-number { background: rgba(255,255,255,0.25); }
    .step-complete .step-number { background: rgba(255,255,255,0.25); }
    .step-inactive .step-number { background: #CBD5E1; color: white; }
    
    /* Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    
    .card-subtitle {
        font-size: 0.9rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    
    /* Candidate card */
    .candidate-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        transition: all 0.2s;
    }
    
    .candidate-card:hover {
        border-color: #A5B4FC;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.08);
        transform: translateY(-2px);
    }
    
    .candidate-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1E293B;
    }
    
    .candidate-title {
        font-size: 0.9rem;
        color: #64748B;
        margin-bottom: 0.75rem;
    }
    
    /* Score badge */
    .score-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 56px;
        height: 56px;
        border-radius: 14px;
        font-size: 1.2rem;
        font-weight: 800;
        color: white;
    }
    
    .score-high { background: linear-gradient(135deg, #10B981, #059669); }
    .score-mid { background: linear-gradient(135deg, #F59E0B, #D97706); }
    .score-low { background: linear-gradient(135deg, #EF4444, #DC2626); }
    
    /* Skill tags */
    .skill-tag {
        display: inline-block;
        background: #EEF2FF;
        color: #4F46E5;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 2px;
    }
    
    .skill-tag-match {
        background: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }
    
    /* AI Summary box */
    .ai-summary {
        background: linear-gradient(135deg, #EEF2FF, #F5F3FF);
        border-left: 3px solid #4F46E5;
        border-radius: 0 12px 12px 0;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.6;
        margin: 0.75rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.82rem;
        color: #94A3B8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Processing steps */
    .process-step {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
        font-size: 0.95rem;
    }
    
    .process-icon-done { color: #10B981; }
    .process-icon-active { color: #4F46E5; }
    .process-icon-pending { color: #CBD5E1; }
    
    /* Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5, #6366F1) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Upload area */
    .stFileUploader > div {
        border-radius: 16px !important;
        border: 2px dashed #A5B4FC !important;
        background: #F8FAFF !important;
    }
    
    .stFileUploader > div:hover {
        border-color: #4F46E5 !important;
        background: #EEF2FF !important;
    }
    
    /* Text inputs */
    .stTextArea textarea, .stTextInput input {
        border-radius: 12px !important;
        border: 1.5px solid #E2E8F0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 12px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #F1F5F9;
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 1.5rem 0;
    }
    
    /* Download button */
    .stDownloadButton > button {
        border-radius: 12px !important;
        background: linear-gradient(135deg, #10B981, #059669) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "step" not in st.session_state:
    st.session_state.step = 1
if "shortlisted" not in st.session_state:
    st.session_state.shortlisted = set()
if "candidates" not in st.session_state:
    st.session_state.candidates = None
if "submission" not in st.session_state:
    st.session_state.submission = None
if "logs" not in st.session_state:
    st.session_state.logs = None
if "elapsed" not in st.session_state:
    st.session_state.elapsed = 0
if "input_file" not in st.session_state:
    st.session_state.input_file = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def render_step_indicators(current_step):
    steps = [
        (1, "Upload Data"),
        (2, "Job Description"),
        (3, "AI Analysis"),
        (4, "Results"),
        (5, "Shortlist"),
    ]
    pills = []
    for num, label in steps:
        if num < current_step:
            cls = "step-complete"
            icon = "✓"
        elif num == current_step:
            cls = "step-active"
            icon = str(num)
        else:
            cls = "step-inactive"
            icon = str(num)
        pills.append(
            f'<div class="step-pill {cls}">'
            f'<span class="step-number">{icon}</span>'
            f'{label}</div>'
        )
    st.markdown(
        f'<div class="steps-container">{"".join(pills)}</div>',
        unsafe_allow_html=True,
    )


def score_to_100(score_0_1):
    """Convert 0-1 score to 0-100 scale."""
    return int(round(score_0_1 * 100))


def score_class(score_100):
    if score_100 >= 70:
        return "score-high"
    elif score_100 >= 50:
        return "score-mid"
    return "score-low"


def load_candidates_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]


def save_uploaded(uploaded_file):
    path = "uploaded_candidates.json"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def run_pipeline_step(step_name, cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def get_candidate_skills(cand, limit=8):
    skills = cand.get("skills", [])
    return [s.get("name", "") for s in skills[:limit]]


def get_strengths(cand, feats_lookup, emb_lookup):
    cid = cand.get("candidate_id", "")
    strengths = []
    feats = feats_lookup.get(cid, {})
    emb = emb_lookup.get(cid, {})
    
    if feats.get("must_have_skill_score", 0) > 0.5:
        strengths.append("Strong match on must-have skills")
    if feats.get("career_depth_score", 0) > 0.4:
        strengths.append("Deep production ML experience")
    if feats.get("title_relevance_score", 0) >= 0.8:
        strengths.append("Directly relevant job title")
    if emb.get("sim_combined", 0) > 0.6:
        strengths.append("High semantic alignment with JD")
    if feats.get("experience_fit_score", 0) > 0.8:
        strengths.append("Ideal experience range")
    if feats.get("product_vs_services_score", 0) > 0.7:
        strengths.append("Product company background")
    if feats.get("notice_period_score", 0) > 0.8:
        strengths.append("Available on short notice")
    
    return strengths[:4] if strengths else ["Meets baseline qualifications"]


def get_concerns(cand, feats_lookup, honey_lookup):
    cid = cand.get("candidate_id", "")
    concerns = []
    feats = feats_lookup.get(cid, {})
    honey = honey_lookup.get(cid, {})
    
    if feats.get("experience_fit_score", 1) < 0.5:
        yoe = cand.get("profile", {}).get("years_of_experience", 0)
        if yoe < 4:
            concerns.append("Below preferred experience range")
        elif yoe > 12:
            concerns.append("May be overqualified for the role")
    if feats.get("location_fit_score", 1) < 0.5:
        concerns.append("Location may not be ideal")
    if feats.get("must_have_skill_score", 1) < 0.2:
        concerns.append("Limited overlap with must-have skills")
    if honey.get("implausibility_score", 0) > 0.1:
        concerns.append("Minor profile inconsistencies detected")
    
    return concerns[:3] if concerns else ["No major concerns identified"]


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown("""
<div class="hero-container">
    <div class="hero-badge">⚡ AI-POWERED RECRUITING</div>
    <div class="hero-title">TalentRank AI</div>
    <div class="hero-subtitle">
        Upload candidates, paste a job description, and let AI find your 
        best matches — ranked with clear explanations, not keyword tricks.
    </div>
</div>
""", unsafe_allow_html=True)

render_step_indicators(st.session_state.step)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Upload Candidate Data
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.step == 1:
    
    col_left, col_spacer, col_right = st.columns([5, 1, 5])
    
    with col_left:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">📤 Upload Candidate Data</div>
            <div class="card-subtitle">
                Upload a JSON or JSONL file containing candidate profiles. 
                Supports up to 200 MB.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded = st.file_uploader(
            "Drag and drop your candidate file",
            type=["json", "jsonl"],
            help="JSON array or JSONL (one candidate per line). Max 200MB.",
        )
        
        if uploaded:
            size_mb = uploaded.size / (1024 * 1024)
            st.success(f"✅ **{uploaded.name}** — {size_mb:.1f} MB")
    
    with col_right:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">📦 Or Try the Demo</div>
            <div class="card-subtitle">
                Don't have a file? Run the pipeline on our built-in 
                50-candidate sample dataset.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        use_demo = st.button("🚀 Use Demo Dataset", use_container_width=True)
    
    st.markdown("---")
    
    # Handle navigation
    if uploaded:
        if st.button("**Analyze Candidates →**", type="primary", use_container_width=True):
            st.session_state.input_file = save_uploaded(uploaded)
            st.session_state.step = 2
            st.rerun()
    
    if use_demo:
        st.session_state.input_file = "sample_candidates.json"
        st.session_state.step = 2
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Job Description
# ═══════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 2:
    
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">📋 Job Description</div>
        <div class="card-subtitle">
            Our system is pre-configured for the Redrob Senior AI Engineer role. 
            Review the key requirements below, then start analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Role Details")
        st.text_input("Job Title", value="Senior AI Engineer — Embeddings & Ranking", disabled=True)
        st.text_input("Experience", value="6–8 years (ideal range)", disabled=True)
        st.text_input("Location", value="Noida / Pune, India (preferred)", disabled=True)
    
    with col2:
        st.markdown("#### 🔧 Must-Have Skills")
        st.markdown("""
        <div style="margin-top: 0.5rem;">
            <span class="skill-tag-match skill-tag">Embeddings</span>
            <span class="skill-tag-match skill-tag">FAISS</span>
            <span class="skill-tag-match skill-tag">Pinecone</span>
            <span class="skill-tag-match skill-tag">Weaviate</span>
            <span class="skill-tag-match skill-tag">PyTorch</span>
            <span class="skill-tag-match skill-tag">TensorFlow</span>
            <span class="skill-tag-match skill-tag">Sentence Transformers</span>
            <span class="skill-tag-match skill-tag">NDCG / MRR</span>
            <span class="skill-tag-match skill-tag">Elasticsearch</span>
            <span class="skill-tag-match skill-tag">Python</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### ✨ Nice-to-Have")
        st.markdown("""
        <div style="margin-top: 0.5rem;">
            <span class="skill-tag">LoRA / QLoRA</span>
            <span class="skill-tag">LambdaMART</span>
            <span class="skill-tag">ONNX / TensorRT</span>
            <span class="skill-tag">RAG</span>
            <span class="skill-tag">Docker / K8s</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_back, col_spacer, col_next = st.columns([1, 3, 1])
    with col_back:
        if st.button("← Back"):
            st.session_state.step = 1
            st.rerun()
    with col_next:
        if st.button("**🧠 Start AI Analysis →**", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: AI Processing
# ═══════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 3:
    
    input_file = st.session_state.input_file
    
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <div class="card-title">🧠 AI Analysis in Progress</div>
        <div class="card-subtitle">
            Analyzing candidate profiles using semantic understanding...
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    pipeline_steps = [
        ("📄 Reading candidate data...", None, 5),
        ("🔍 Extracting 14 structured features...", [
            sys.executable, "precompute/precompute_features.py",
            "--input", input_file, "--output", "artifacts/features.parquet",
        ], 25),
        ("🛡️ Running honeypot detection (7 rules)...", [
            sys.executable, "precompute/precompute_honeypot_flags.py",
            "--input", input_file, "--output", "artifacts/honeypot_flags.parquet",
        ], 50),
        ("🧠 Computing semantic embeddings (MiniLM-L6)...", [
            sys.executable, "precompute/precompute_embeddings.py",
            "--input", input_file, "--output", "artifacts/embeddings.parquet",
        ], 80),
        ("📊 Ranking candidates & generating explanations...", None, 90),
    ]
    
    logs = {}
    all_ok = True
    
    for step_label, cmd, progress in pipeline_steps:
        status_text.markdown(f"**{step_label}**")
        progress_bar.progress(progress)
        
        if cmd:
            log = run_pipeline_step(step_label, cmd)
            logs[step_label] = log
            if log["returncode"] != 0:
                st.error(f"❌ Failed: {step_label}")
                st.code(log["stderr"][-2000:])
                all_ok = False
                break
        else:
            time.sleep(0.5)
    
    if all_ok:
        # Run ranking
        candidates = load_candidates_from_file(input_file)
        top_n = min(100, len(candidates))
        
        status_text.markdown("**🏆 Generating final rankings...**")
        
        rank_cmd = [
            sys.executable, "rank.py",
            "--input", input_file,
            "--output", "submission_demo.csv",
            "--top-n", str(top_n),
        ]
        t0 = time.time()
        rank_result = subprocess.run(rank_cmd, capture_output=True, text=True, cwd=".")
        elapsed = time.time() - t0
        
        if rank_result.returncode == 0:
            progress_bar.progress(100)
            status_text.markdown("**✅ Analysis complete!**")
            
            # Store results
            st.session_state.candidates = candidates
            st.session_state.submission = pd.read_csv("submission_demo.csv")
            st.session_state.logs = logs
            st.session_state.elapsed = elapsed
            
            time.sleep(1)
            st.session_state.step = 4
            st.rerun()
        else:
            st.error("Ranking failed!")
            st.code(rank_result.stderr[-2000:])


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Results Dashboard
# ═══════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 4:
    
    submission = st.session_state.submission
    candidates = st.session_state.candidates
    elapsed = st.session_state.elapsed
    candidate_lookup = {c["candidate_id"]: c for c in candidates}
    
    # Load feature data for strengths/concerns
    features_df = pd.read_parquet("artifacts/features.parquet")
    honeypot_df = pd.read_parquet("artifacts/honeypot_flags.parquet")
    embeddings_df = pd.read_parquet("artifacts/embeddings.parquet")
    feats_lookup = features_df.set_index("candidate_id").to_dict("index")
    honey_lookup = honeypot_df.set_index("candidate_id").to_dict("index")
    emb_lookup = embeddings_df.set_index("candidate_id").to_dict("index")
    
    excluded = (honeypot_df["implausibility_score"] >= 0.3).sum()
    
    # --- Metrics row ---
    m1, m2, m3, m4, m5 = st.columns(5)
    
    for col, val, label in [
        (m1, len(candidates), "CANDIDATES"),
        (m2, int(excluded), "HONEYPOTS"),
        (m3, len(submission), "RANKED"),
        (m4, f"{elapsed:.1f}s", "RUNTIME"),
        (m5, f"{score_to_100(submission['score'].max())}", "TOP SCORE"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- Tabs ---
    tab_all, tab_short, tab_logs = st.tabs(["📊 All Candidates", "⭐ Shortlisted", "📋 Logs"])
    
    with tab_all:
        # Filters
        col_search, col_min_score, col_exp = st.columns([3, 1, 1])
        with col_search:
            search = st.text_input("🔍 Search candidates...", placeholder="Search by name, title, skills...")
        with col_min_score:
            min_score = st.slider("Min Score", 0, 100, 0)
        with col_exp:
            exp_filter = st.selectbox("Experience", ["All", "0-3y", "4-6y", "7-10y", "10+y"])
        
        # Filter logic
        filtered = submission.copy()
        
        for idx, row in filtered.iterrows():
            cand = candidate_lookup.get(row["candidate_id"], {})
            profile = cand.get("profile", {})
            filtered.loc[idx, "_title"] = profile.get("current_title", "")
            filtered.loc[idx, "_yoe"] = profile.get("years_of_experience", 0)
            filtered.loc[idx, "_location"] = profile.get("location", "")
            filtered.loc[idx, "_company"] = profile.get("current_company", "")
            filtered.loc[idx, "_score100"] = score_to_100(row["score"])
        
        if search:
            search_lower = search.lower()
            mask = (
                filtered["candidate_id"].str.lower().str.contains(search_lower, na=False) |
                filtered["_title"].str.lower().str.contains(search_lower, na=False) |
                filtered["reasoning"].str.lower().str.contains(search_lower, na=False) |
                filtered["_company"].str.lower().str.contains(search_lower, na=False)
            )
            filtered = filtered[mask]
        
        if min_score > 0:
            filtered = filtered[filtered["_score100"] >= min_score]
        
        if exp_filter != "All":
            ranges = {"0-3y": (0, 3), "4-6y": (4, 6), "7-10y": (7, 10), "10+y": (10, 50)}
            lo, hi = ranges[exp_filter]
            filtered = filtered[(filtered["_yoe"] >= lo) & (filtered["_yoe"] <= hi)]
        
        st.caption(f"Showing {len(filtered)} of {len(submission)} candidates")
        
        # Render candidate cards
        for _, row in filtered.iterrows():
            cid = row["candidate_id"]
            cand = candidate_lookup.get(cid, {})
            profile = cand.get("profile", {})
            score_100 = score_to_100(row["score"])
            s_class = score_class(score_100)
            skills = get_candidate_skills(cand)
            strengths = get_strengths(cand, feats_lookup, emb_lookup)
            concerns = get_concerns(cand, feats_lookup, honey_lookup)
            
            skills_html = " ".join(f'<span class="skill-tag">{s}</span>' for s in skills)
            
            is_shortlisted = cid in st.session_state.shortlisted
            
            with st.container():
                c1, c2 = st.columns([1, 12])
                
                with c1:
                    st.markdown(f"""
                    <div class="score-badge {s_class}" style="margin-top: 0.5rem;">
                        {score_100}
                    </div>
                    """, unsafe_allow_html=True)
                
                with c2:
                    st.markdown(f"""
                    <div class="candidate-name">{cid} — {profile.get('current_title', 'N/A')}</div>
                    <div class="candidate-title">
                        {profile.get('years_of_experience', 0):.0f}y experience · 
                        {profile.get('current_company', 'N/A')} · 
                        {profile.get('location', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f'<div class="ai-summary">💡 {row["reasoning"]}</div>', unsafe_allow_html=True)
                    
                    exp_key = f"exp_{cid}"
                    with st.expander("View details"):
                        dc1, dc2 = st.columns(2)
                        with dc1:
                            st.markdown("**✅ Strengths**")
                            for s in strengths:
                                st.markdown(f"- {s}")
                        with dc2:
                            st.markdown("**⚠️ Potential Concerns**")
                            for c in concerns:
                                st.markdown(f"- {c}")
                        
                        st.markdown(f"**Skills:** {skills_html}", unsafe_allow_html=True)
                        
                        assessments = cand.get("redrob_signals", {}).get("skill_assessment_scores", {})
                        if assessments:
                            st.markdown("**Redrob Assessments:**")
                            for skill, sc in sorted(assessments.items(), key=lambda x: x[1], reverse=True)[:5]:
                                st.markdown(f"- {skill}: **{sc:.0f}**/100")
                    
                    # Shortlist button
                    btn_label = "✓ Shortlisted" if is_shortlisted else "⭐ Shortlist"
                    if st.button(btn_label, key=f"btn_{cid}"):
                        if is_shortlisted:
                            st.session_state.shortlisted.discard(cid)
                        else:
                            st.session_state.shortlisted.add(cid)
                        st.rerun()
                
                st.markdown("---")
    
    with tab_short:
        shortlisted = st.session_state.shortlisted
        
        if not shortlisted:
            st.info("No candidates shortlisted yet. Click **⭐ Shortlist** on any candidate card.")
        else:
            st.markdown(f"### ⭐ {len(shortlisted)} Shortlisted Candidates")
            
            short_rows = []
            for _, row in submission.iterrows():
                if row["candidate_id"] in shortlisted:
                    cand = candidate_lookup.get(row["candidate_id"], {})
                    profile = cand.get("profile", {})
                    short_rows.append({
                        "Rank": int(row["rank"]),
                        "Candidate ID": row["candidate_id"],
                        "Score": score_to_100(row["score"]),
                        "Title": profile.get("current_title", ""),
                        "Experience": f"{profile.get('years_of_experience', 0):.0f}y",
                        "Location": profile.get("location", ""),
                        "Recommendation": row["reasoning"][:150] + "...",
                    })
            
            short_df = pd.DataFrame(short_rows)
            st.dataframe(short_df, use_container_width=True, hide_index=True)
            
            # Export
            csv_data = short_df.to_csv(index=False)
            st.download_button(
                "📥 Export Shortlist as CSV",
                data=csv_data,
                file_name="shortlisted_candidates.csv",
                mime="text/csv",
                use_container_width=True,
            )
    
    with tab_logs:
        if st.session_state.logs:
            for step_name, log in st.session_state.logs.items():
                with st.expander(step_name):
                    stdout = log["stdout"]
                    st.code(stdout[-3000:] if len(stdout) > 3000 else stdout)
    
    # Bottom nav
    st.markdown("---")
    col_back, col_spacer, col_restart = st.columns([1, 4, 1])
    with col_back:
        if st.button("← New Analysis"):
            for key in ["step", "candidates", "submission", "logs", "elapsed", "input_file"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 1
            st.session_state.shortlisted = set()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 redirect (just show tab_short from step 4)
# ═══════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 5:
    st.session_state.step = 4
    st.rerun()
