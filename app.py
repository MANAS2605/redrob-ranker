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
    
    /* Global font override — only target text elements, not internal Streamlit */
    .stApp, .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, span, div, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ─── Hero ─── */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 0.5rem;
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
        margin-bottom: 0.75rem;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #64748B;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto 1.5rem;
        line-height: 1.6;
    }
    
    /* ─── Step indicators ─── */
    .steps-container {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin: 1rem 0 1.5rem;
        flex-wrap: wrap;
    }
    
    .step-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 18px;
        border-radius: 24px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    
    .step-active {
        background: linear-gradient(135deg, #4F46E5, #6366F1);
        color: white;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
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
        font-size: 0.72rem;
        font-weight: 700;
    }
    
    .step-active .step-number { background: rgba(255,255,255,0.25); }
    .step-complete .step-number { background: rgba(255,255,255,0.25); }
    .step-inactive .step-number { background: #CBD5E1; color: white; }
    
    /* ─── Glass card (visual-only header) ─── */
    .glass-card {
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.4rem;
    }
    
    .card-subtitle {
        font-size: 0.88rem;
        color: #64748B;
        line-height: 1.5;
    }
    
    /* ─── Candidate card ─── */
    .cand-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.25rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    
    .cand-header {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        margin-bottom: 0.5rem;
    }
    
    .cand-score {
        flex-shrink: 0;
        width: 52px;
        height: 52px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        font-weight: 800;
        color: white;
    }
    
    .cand-score-high { background: linear-gradient(135deg, #10B981, #059669); }
    .cand-score-mid  { background: linear-gradient(135deg, #F59E0B, #D97706); }
    .cand-score-low  { background: linear-gradient(135deg, #EF4444, #DC2626); }
    
    .cand-info { flex: 1; min-width: 0; }
    
    .cand-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 2px;
        word-wrap: break-word;
    }
    
    .cand-meta {
        font-size: 0.85rem;
        color: #64748B;
        margin-bottom: 0.5rem;
    }
    
    .cand-meta span {
        margin-right: 6px;
    }
    
    /* ─── AI summary box ─── */
    .ai-box {
        background: linear-gradient(135deg, #EEF2FF, #F5F3FF);
        border-left: 3px solid #4F46E5;
        border-radius: 0 12px 12px 0;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #334155;
        line-height: 1.55;
        margin: 0.5rem 0 0.75rem;
        word-wrap: break-word;
    }
    
    /* ─── Skill tags ─── */
    .skill-tag {
        display: inline-block;
        background: #EEF2FF;
        color: #4F46E5;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 2px;
    }
    
    .skill-match {
        background: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }
    
    /* ─── Metric cards ─── */
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 1.1rem 0.75rem;
        text-align: center;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }
    
    /* ─── Buttons ─── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
    }
    
    /* ─── Upload area ─── */
    [data-testid="stFileUploader"] > div {
        border-radius: 14px !important;
    }
    
    /* ─── Text inputs ─── */
    .stTextArea textarea, .stTextInput input {
        border-radius: 10px !important;
    }
    
    /* ─── Download button ─── */
    .stDownloadButton > button {
        border-radius: 10px !important;
        background: linear-gradient(135deg, #10B981, #059669) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }
    
    /* ─── Details section ─── */
    .detail-section {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    
    .detail-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 6px;
    }
    
    .detail-item {
        font-size: 0.88rem;
        color: #334155;
        padding: 2px 0;
    }
    
    .strength-item { color: #059669; }
    .concern-item  { color: #D97706; }
    
    /* ─── Divider ─── */
    hr {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 1.25rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key, default in [
    ("step", 1),
    ("shortlisted", set()),
    ("candidates", None),
    ("submission", None),
    ("logs", None),
    ("elapsed", 0),
    ("input_file", None),
    ("jd_mode", "preset"),      # "preset" or "custom"
    ("custom_jd_text", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_steps(current):
    labels = ["Upload Data", "Job Description", "AI Analysis", "Results", "Shortlist"]
    pills = []
    for i, label in enumerate(labels, 1):
        if i < current:
            pills.append(f'<div class="step-pill step-complete"><span class="step-number">✓</span>{label}</div>')
        elif i == current:
            pills.append(f'<div class="step-pill step-active"><span class="step-number">{i}</span>{label}</div>')
        else:
            pills.append(f'<div class="step-pill step-inactive"><span class="step-number">{i}</span>{label}</div>')
    st.markdown(f'<div class="steps-container">{"".join(pills)}</div>', unsafe_allow_html=True)


def score100(s):
    return int(round(s * 100))


def score_cls(s):
    if s >= 70: return "cand-score-high"
    if s >= 50: return "cand-score-mid"
    return "cand-score-low"


def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        c = f.read(1); f.seek(0)
        return json.load(f) if c == "[" else [json.loads(l) for l in f if l.strip()]


def run_step(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    return {"stdout": r.stdout, "stderr": r.stderr, "rc": r.returncode}


def get_skills(cand, n=10):
    return [s.get("name", "") for s in cand.get("skills", [])[:n] if s.get("name")]


def get_education(cand):
    edu = cand.get("education", [])
    lines = []
    for e in edu[:3]:
        deg = e.get("degree", "")
        field = e.get("field_of_study", "")
        inst = e.get("institution", "")
        yr = e.get("graduation_year", "")
        parts = [p for p in [deg, field] if p]
        line = " in ".join(parts) if parts else "Degree"
        if inst:
            line += f" — {inst}"
        if yr:
            line += f" ({yr})"
        lines.append(line)
    return lines if lines else ["Not specified"]


def get_strengths(cand, feats, embs):
    cid = cand.get("candidate_id", "")
    f = feats.get(cid, {})
    e = embs.get(cid, {})
    s = []
    if f.get("must_have_skill_score", 0) > 0.5:
        s.append("Strong match on must-have skills")
    if f.get("career_depth_score", 0) > 0.4:
        s.append("Deep production ML/search experience")
    if f.get("title_relevance_score", 0) >= 0.8:
        s.append("Directly relevant job title")
    if e.get("sim_combined", 0) > 0.6:
        s.append("High semantic alignment with JD")
    if f.get("experience_fit_score", 0) > 0.8:
        s.append("Ideal experience range (6-8 years)")
    if f.get("product_vs_services_score", 0) > 0.7:
        s.append("Product company background")
    if f.get("notice_period_score", 0) > 0.8:
        s.append("Available on short notice")
    if f.get("github_score", 0) > 0.6:
        s.append("Active GitHub profile")
    return s[:5] if s else ["Meets baseline qualifications"]


def get_concerns(cand, feats, honeys):
    cid = cand.get("candidate_id", "")
    f = feats.get(cid, {})
    h = honeys.get(cid, {})
    c = []
    yoe = cand.get("profile", {}).get("years_of_experience", 0)
    if f.get("experience_fit_score", 1) < 0.5:
        if yoe < 4:
            c.append("Below preferred experience range")
        elif yoe > 12:
            c.append("May be overqualified for this role")
    if f.get("location_fit_score", 1) < 0.5:
        loc = cand.get("profile", {}).get("location", "Unknown")
        c.append(f"Location ({loc}) may require relocation")
    if f.get("must_have_skill_score", 1) < 0.2:
        c.append("Limited overlap with must-have skills")
    if f.get("notice_period_score", 1) < 0.4:
        c.append("Long notice period")
    if h.get("implausibility_score", 0) > 0.1:
        c.append("Minor profile inconsistencies noted")
    return c[:4] if c else ["No major concerns identified"]


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

render_steps(st.session_state.step)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Upload
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.step == 1:

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">📤 Upload Candidate Data</div>
            <div class="card-subtitle">
                Upload a JSON or JSONL file containing candidate profiles.
                Supports files up to 200 MB.
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drag and drop your candidate file here",
            type=["json", "jsonl"],
            help="Accepted formats: JSON array or JSONL (one candidate per line).",
        )

        if uploaded:
            sz = uploaded.size / (1024 * 1024)
            st.success(f"✅ **{uploaded.name}** — {sz:.1f} MB ready")

            if st.button("**Analyze Candidates →**", type="primary", use_container_width=True):
                path = "uploaded_candidates.json"
                with open(path, "wb") as f:
                    f.write(uploaded.getbuffer())
                st.session_state.input_file = path
                st.session_state.step = 2
                st.rerun()

    with col_r:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">📦 Try the Demo</div>
            <div class="card-subtitle">
                No file? Run the pipeline on our built-in 50-candidate 
                sample dataset to see TalentRank AI in action.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Use Demo Dataset (50 candidates)", use_container_width=True):
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
            Choose the pre-configured Senior AI Engineer role, or paste your own 
            job description to rank candidates against any role.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_preset, tab_custom = st.tabs(["🎯 Pre-configured JD (Senior AI Engineer)", "✏️ Paste Custom JD"])

    with tab_preset:
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("#### 🎯 Role Details")
            st.markdown("""
            | Field | Value |
            |-------|-------|
            | **Job Title** | Senior AI Engineer — Embeddings & Ranking |
            | **Ideal Experience** | 6–8 years |
            | **Preferred Location** | Noida / Pune, India |
            | **Company Type** | Product company preferred |
            | **Notice Period** | ≤ 30 days ideal |
            """)

        with col2:
            st.markdown("#### 🔧 Must-Have Skills")
            st.markdown("""
            <div style="margin: 0.5rem 0;">
                <span class="skill-match skill-tag">Embeddings</span>
                <span class="skill-match skill-tag">FAISS</span>
                <span class="skill-match skill-tag">Pinecone</span>
                <span class="skill-match skill-tag">Weaviate</span>
                <span class="skill-match skill-tag">PyTorch</span>
                <span class="skill-match skill-tag">TensorFlow</span>
                <span class="skill-match skill-tag">Sentence Transformers</span>
                <span class="skill-match skill-tag">NDCG / MRR</span>
                <span class="skill-match skill-tag">Elasticsearch</span>
                <span class="skill-match skill-tag">Python</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### ✨ Nice-to-Have Skills")
            st.markdown("""
            <div style="margin: 0.5rem 0;">
                <span class="skill-tag">LoRA / QLoRA</span>
                <span class="skill-tag">LambdaMART</span>
                <span class="skill-tag">ONNX / TensorRT</span>
                <span class="skill-tag">RAG</span>
                <span class="skill-tag">Docker / K8s</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🧠 **Analyze with Pre-configured JD →**", type="primary", use_container_width=True, key="btn_preset"):
            st.session_state.jd_mode = "preset"
            st.session_state.custom_jd_text = ""
            st.session_state.step = 3
            st.rerun()

    with tab_custom:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">✏️ Paste Your Job Description</div>
            <div class="card-subtitle">
                Paste the full job description text below. The AI will create 
                semantic embeddings from your JD to rank candidates against 
                your specific role requirements.
            </div>
        </div>
        """, unsafe_allow_html=True)

        custom_jd = st.text_area(
            "Job Description",
            height=300,
            placeholder=(
                "Paste your full job description here…\n\n"
                "Example:\n"
                "We are looking for a Senior Backend Engineer with 5+ years of experience "
                "in Python, Django, and distributed systems. The ideal candidate has experience "
                "with microservices architecture, PostgreSQL, Redis, and AWS…"
            ),
            help="Include role title, required skills, experience, and responsibilities for best results.",
        )

        if custom_jd and len(custom_jd.strip()) > 50:
            st.success(f"✅ JD loaded — {len(custom_jd.strip())} characters")
            st.caption("💡 The AI will create 3 semantic persona variants from your JD to match candidates.")

            if st.button("🧠 **Analyze with Custom JD →**", type="primary", use_container_width=True, key="btn_custom"):
                st.session_state.jd_mode = "custom"
                st.session_state.custom_jd_text = custom_jd.strip()
                st.session_state.step = 3
                st.rerun()
        elif custom_jd:
            st.warning("Please enter at least 50 characters for meaningful analysis.")

    st.markdown("---")

    if st.button("← Back to Upload"):
        st.session_state.step = 1
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: AI Processing
# ═══════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 3:

    inp = st.session_state.input_file

    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <div class="card-title">🧠 AI Analysis in Progress</div>
        <div class="card-subtitle">
            Analyzing candidate profiles using semantic understanding…
        </div>
    </div>
    """, unsafe_allow_html=True)

    bar = st.progress(0)
    msg = st.empty()

    steps = [
        ("📄 Reading candidate data…", None, 5),
        ("🔍 Extracting 14 structured features…", [
            sys.executable, "precompute/precompute_features.py",
            "--input", inp, "--output", "artifacts/features.parquet",
        ], 25),
        ("🛡️ Running honeypot detection (7 plausibility rules)…", [
            sys.executable, "precompute/precompute_honeypot_flags.py",
            "--input", inp, "--output", "artifacts/honeypot_flags.parquet",
        ], 50),
        ("🧠 Computing semantic embeddings (MiniLM-L6-v2)…", None, 75),  # placeholder, built dynamically below
    ]

    # Build the embeddings command dynamically (may include --jd-file)
    emb_cmd = [
        sys.executable, "precompute/precompute_embeddings.py",
        "--input", inp, "--output", "artifacts/embeddings.parquet",
    ]
    jd_file_path = None
    if st.session_state.jd_mode == "custom" and st.session_state.custom_jd_text:
        jd_file_path = "custom_jd.txt"
        with open(jd_file_path, "w", encoding="utf-8") as f:
            f.write(st.session_state.custom_jd_text)
        emb_cmd.extend(["--jd-file", jd_file_path])

    # Replace the placeholder embeddings step with the actual command
    steps[-1] = ("🧠 Computing semantic embeddings (MiniLM-L6-v2)…", emb_cmd, 80)

    # Add final placeholder step
    steps.append(
        ("📊 Ranking candidates & generating explanations…", None, 90),
    )

    logs = {}
    ok = True

    for label, cmd, pct in steps:
        msg.markdown(f"**{label}**")
        bar.progress(pct)
        if cmd:
            log = run_step(cmd)
            logs[label] = log
            if log["rc"] != 0:
                st.error(f"❌ {label}")
                st.code(log["stderr"][-3000:])
                ok = False
                break
        else:
            time.sleep(0.5)

    if ok:
        candidates = load_file(inp)
        top_n = min(100, len(candidates))

        msg.markdown("**🏆 Generating final rankings…**")

        t0 = time.time()
        r = subprocess.run([
            sys.executable, "rank.py",
            "--input", inp,
            "--output", "submission_demo.csv",
            "--top-n", str(top_n),
        ], capture_output=True, text=True, cwd=".")
        elapsed = time.time() - t0

        if r.returncode == 0:
            bar.progress(100)
            msg.markdown("**✅ Analysis complete!**")

            st.session_state.candidates = candidates
            st.session_state.submission = pd.read_csv("submission_demo.csv")
            st.session_state.logs = logs
            st.session_state.elapsed = elapsed

            time.sleep(0.8)
            st.session_state.step = 4
            st.rerun()
        else:
            st.error("❌ Ranking failed")
            st.code(r.stderr[-3000:])


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Results Dashboard
# ═══════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 4:

    sub = st.session_state.submission
    cands = st.session_state.candidates
    elapsed = st.session_state.elapsed
    c_lookup = {c["candidate_id"]: c for c in cands}

    # Load enrichment data
    feat_df = pd.read_parquet("artifacts/features.parquet")
    honey_df = pd.read_parquet("artifacts/honeypot_flags.parquet")
    emb_df = pd.read_parquet("artifacts/embeddings.parquet")
    feat_lu = feat_df.set_index("candidate_id").to_dict("index")
    honey_lu = honey_df.set_index("candidate_id").to_dict("index")
    emb_lu = emb_df.set_index("candidate_id").to_dict("index")
    excluded = int((honey_df["implausibility_score"] >= 0.3).sum())
    top_score = score100(sub["score"].max())

    # ── Metrics ──
    cols = st.columns(5)
    metrics = [
        (len(cands), "CANDIDATES PROCESSED"),
        (excluded, "HONEYPOTS EXCLUDED"),
        (len(sub), "CANDIDATES RANKED"),
        (f"{elapsed:.1f}s", "RANKING TIME"),
        (top_score, "TOP AI SCORE"),
    ]
    for col, (val, label) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabs ──
    tab_all, tab_short, tab_logs = st.tabs([
        f"📊 All Candidates ({len(sub)})",
        f"⭐ Shortlisted ({len(st.session_state.shortlisted)})",
        "📋 Pipeline Logs",
    ])

    # ── ALL CANDIDATES TAB ──
    with tab_all:

        # Filters row
        fc1, fc2, fc3 = st.columns([3, 1, 1])
        with fc1:
            search = st.text_input(
                "🔍 Search candidates",
                placeholder="Search by ID, title, company, skills…",
                label_visibility="collapsed",
            )
        with fc2:
            min_score = st.slider("Min Score", 0, 100, 0, help="Filter by minimum AI score")
        with fc3:
            exp_filter = st.selectbox("Experience", ["All", "0-3y", "4-6y", "7-10y", "10+y"])

        # Build enriched DataFrame (vectorized)
        enriched = sub.copy()
        enriched["_title"] = enriched["candidate_id"].map(
            lambda cid: c_lookup.get(cid, {}).get("profile", {}).get("current_title", "")
        )
        enriched["_yoe"] = enriched["candidate_id"].map(
            lambda cid: c_lookup.get(cid, {}).get("profile", {}).get("years_of_experience", 0)
        )
        enriched["_company"] = enriched["candidate_id"].map(
            lambda cid: c_lookup.get(cid, {}).get("profile", {}).get("current_company", "")
        )
        enriched["_location"] = enriched["candidate_id"].map(
            lambda cid: c_lookup.get(cid, {}).get("profile", {}).get("location", "")
        )
        enriched["_s100"] = enriched["score"].apply(score100)

        # Apply filters
        if search:
            q = search.lower()
            mask = (
                enriched["candidate_id"].str.lower().str.contains(q, na=False) |
                enriched["_title"].str.lower().str.contains(q, na=False) |
                enriched["_company"].str.lower().str.contains(q, na=False) |
                enriched["reasoning"].str.lower().str.contains(q, na=False)
            )
            enriched = enriched[mask]

        if min_score > 0:
            enriched = enriched[enriched["_s100"] >= min_score]

        if exp_filter != "All":
            ranges = {"0-3y": (0, 3), "4-6y": (4, 6), "7-10y": (7, 10), "10+y": (10, 99)}
            lo, hi = ranges[exp_filter]
            enriched = enriched[(enriched["_yoe"] >= lo) & (enriched["_yoe"] <= hi)]

        st.caption(f"Showing **{len(enriched)}** of {len(sub)} candidates")

        # ── Download ALL results button ──
        all_csv_rows = []
        for _, row in sub.iterrows():
            c = c_lookup.get(row["candidate_id"], {})
            p = c.get("profile", {})
            all_csv_rows.append({
                "Rank": int(row["rank"]),
                "Candidate ID": row["candidate_id"],
                "AI Score": score100(row["score"]),
                "Title": p.get("current_title", ""),
                "Experience (years)": p.get("years_of_experience", 0),
                "Company": p.get("current_company", ""),
                "Location": p.get("location", ""),
                "Country": p.get("country", ""),
                "Skills": ", ".join(get_skills(c, 15)),
                "AI Reasoning": row["reasoning"],
            })
        all_csv_df = pd.DataFrame(all_csv_rows)
        st.download_button(
            "📥 Download All Results as CSV",
            data=all_csv_df.to_csv(index=False),
            file_name="talentrank_all_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("")

        # ── Render candidate cards ──
        for _, row in enriched.iterrows():
            cid = row["candidate_id"]
            cand = c_lookup.get(cid, {})
            profile = cand.get("profile", {})
            s = score100(row["score"])
            cls = score_cls(s)
            skills = get_skills(cand)
            education = get_education(cand)
            strengths = get_strengths(cand, feat_lu, emb_lu)
            concerns = get_concerns(cand, feat_lu, honey_lu)
            is_short = cid in st.session_state.shortlisted

            # Skills HTML
            skills_html = " ".join(f'<span class="skill-tag">{sk}</span>' for sk in skills)

            # Card header with score + info
            st.markdown(f"""
            <div class="cand-card">
                <div class="cand-header">
                    <div class="cand-score {cls}">{s}</div>
                    <div class="cand-info">
                        <div class="cand-name">#{int(row['rank'])} — {profile.get('current_title', 'N/A')}</div>
                        <div class="cand-meta">
                            <span>🆔 {cid}</span> ·
                            <span>📅 {profile.get('years_of_experience', 0):.0f}y experience</span> ·
                            <span>🏢 {profile.get('current_company', 'N/A')}</span> ·
                            <span>📍 {profile.get('location', 'N/A')}, {profile.get('country', '')}</span>
                        </div>
                    </div>
                </div>
                <div class="ai-box">💡 {row['reasoning']}</div>
                <div>{skills_html}</div>
            </div>
            """, unsafe_allow_html=True)

            # Expandable details
            with st.expander(f"View full profile — {cid}"):
                d1, d2, d3 = st.columns(3)

                with d1:
                    st.markdown("**✅ Strengths**")
                    for item in strengths:
                        st.markdown(f"- 🟢 {item}")

                with d2:
                    st.markdown("**⚠️ Potential Concerns**")
                    for item in concerns:
                        st.markdown(f"- 🟡 {item}")

                with d3:
                    st.markdown("**🎓 Education**")
                    for item in education:
                        st.markdown(f"- {item}")

                # Assessments
                assessments = cand.get("redrob_signals", {}).get("skill_assessment_scores", {})
                if assessments:
                    st.markdown("**📊 Redrob Assessment Scores**")
                    a_cols = st.columns(min(len(assessments), 5))
                    for i, (skill, sc) in enumerate(sorted(assessments.items(), key=lambda x: x[1], reverse=True)[:5]):
                        with a_cols[i]:
                            st.metric(skill, f"{sc:.0f}/100")

                # Additional profile info
                notice = profile.get("notice_period_days")
                industry = profile.get("current_industry", "")
                if notice or industry:
                    st.markdown("**📋 Additional Info**")
                    info_parts = []
                    if notice is not None:
                        info_parts.append(f"Notice period: {notice} days")
                    if industry:
                        info_parts.append(f"Industry: {industry}")
                    st.markdown(" · ".join(info_parts))

            # Shortlist button
            lbl = "✅ Shortlisted" if is_short else "⭐ Add to Shortlist"
            if st.button(lbl, key=f"sl_{cid}", use_container_width=False):
                if is_short:
                    st.session_state.shortlisted.discard(cid)
                else:
                    st.session_state.shortlisted.add(cid)
                st.rerun()

            st.markdown("")  # Spacer

    # ── SHORTLIST TAB ──
    with tab_short:
        shortlist = st.session_state.shortlisted

        if not shortlist:
            st.info("No candidates shortlisted yet. Click **⭐ Add to Shortlist** on any candidate to add them here.")
        else:
            st.markdown(f"### ⭐ {len(shortlist)} Shortlisted Candidates")

            rows = []
            for _, row in sub.iterrows():
                if row["candidate_id"] in shortlist:
                    c = c_lookup.get(row["candidate_id"], {})
                    p = c.get("profile", {})
                    rows.append({
                        "Rank": int(row["rank"]),
                        "Candidate ID": row["candidate_id"],
                        "AI Score": score100(row["score"]),
                        "Title": p.get("current_title", ""),
                        "Experience": f"{p.get('years_of_experience', 0):.0f}y",
                        "Company": p.get("current_company", ""),
                        "Location": p.get("location", ""),
                        "Skills": ", ".join(get_skills(c, 8)),
                        "Recommendation": row["reasoning"][:200],
                    })

            sdf = pd.DataFrame(rows)
            st.dataframe(sdf, use_container_width=True, hide_index=True)

            st.download_button(
                "📥 Export Shortlist as CSV",
                data=sdf.to_csv(index=False),
                file_name="shortlisted_candidates.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── LOGS TAB ──
    with tab_logs:
        if st.session_state.logs:
            for name, log in st.session_state.logs.items():
                with st.expander(name):
                    out = log["stdout"]
                    st.code(out[-3000:] if len(out) > 3000 else out)
        else:
            st.info("No pipeline logs available.")

    # ── Bottom nav ──
    st.markdown("---")
    if st.button("🔄 Start New Analysis"):
        for k in ["step", "candidates", "submission", "logs", "elapsed", "input_file", "jd_mode", "custom_jd_text"]:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.step = 1
        st.session_state.shortlisted = set()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 → redirect to Results tab
# ═══════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 5:
    st.session_state.step = 4
    st.rerun()
