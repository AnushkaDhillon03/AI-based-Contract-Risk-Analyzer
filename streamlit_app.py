"""
AI Contract Risk Analyzer — Streamlit Frontend
Connects to the FastAPI backend at http://127.0.0.1:8080/api/v1
"""

import io
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Contract Risk Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_BASE = "https://ai-based-contract-risk-analyzer.onrender.com/api/v1"

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Fonts ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Root variables ───────────────────────────────────── */
:root {
    --navy:    #0F1E36;
    --slate:   #1C3358;
    --blue:    #2A5C99;
    --accent:  #3B82F6;
    --gold:    #E8B84B;
    --surface: #F7F9FC;
    --card:    #FFFFFF;
    --text:    #1A202C;
    --muted:   #64748B;
    --border:  #E2E8F0;
    --red:     #DC2626;
    --amber:   #D97706;
    --green:   #16A34A;
    --red-bg:  #FEF2F2;
    --amber-bg:#FFFBEB;
    --green-bg:#F0FDF4;
}

/* ── Global resets ────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}
.main { background: var(--surface); }
.block-container { padding: 0 2rem 4rem 2rem !important; max-width: 1300px; }

/* ── Header ───────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, var(--navy) 0%, var(--slate) 60%, #1e4a7a 100%);
    border-radius: 0 0 24px 24px;
    padding: 2.5rem 3rem;
    margin: -1rem -2rem 2.5rem -2rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.app-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(232,184,75,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.logo-mark {
    width: 68px; height: 68px;
    background: linear-gradient(135deg, var(--gold), #f59e0b);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem;
    box-shadow: 0 8px 24px rgba(232,184,75,0.35);
    flex-shrink: 0;
    position: relative; z-index: 1;
}
.header-text { position: relative; z-index: 1; }
.header-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.1rem;
    color: #FFFFFF;
    margin: 0;
    line-height: 1.15;
    letter-spacing: -0.5px;
}
.header-subtitle {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.65);
    margin: 0.4rem 0 0 0;
    font-weight: 400;
    max-width: 560px;
    line-height: 1.5;
}
.header-badge {
    margin-left: auto;
    position: relative; z-index: 1;
    background: rgba(59,130,246,0.2);
    border: 1px solid rgba(59,130,246,0.35);
    color: #93C5FD;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    white-space: nowrap;
}

/* ── Section headings ─────────────────────────────────── */
.section-heading {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--navy);
    margin: 2rem 0 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid var(--border);
    letter-spacing: -0.2px;
}
.section-heading .icon { font-size: 1.25rem; }

/* ── Card containers ──────────────────────────────────── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.card-flat {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
}

/* ── Input section ────────────────────────────────────── */
.input-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.5rem;
}
.divider-or {
    display: flex;
    align-items: center;
    gap: 1rem;
    color: var(--muted);
    font-size: 0.85rem;
    font-weight: 500;
    margin: 1rem 0;
}
.divider-or::before, .divider-or::after {
    content: ''; flex: 1; height: 1px; background: var(--border);
}

/* ── Risk severity badges ─────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-high   { background: var(--red-bg);   color: var(--red);   border: 1px solid #FECACA; }
.badge-medium { background: var(--amber-bg); color: var(--amber); border: 1px solid #FDE68A; }
.badge-low    { background: var(--green-bg); color: var(--green); border: 1px solid #BBF7D0; }

/* ── Risk rows ────────────────────────────────────────── */
.risk-row {
    display: flex; align-items: center; gap: 1rem;
    padding: 0.9rem 1.1rem;
    border-radius: 10px;
    margin-bottom: 0.6rem;
    border: 1px solid transparent;
}
.risk-row.high   { background: var(--red-bg);   border-color: #FECACA; }
.risk-row.medium { background: var(--amber-bg); border-color: #FDE68A; }
.risk-row.low    { background: var(--green-bg); border-color: #BBF7D0; }
.risk-row .risk-icon { font-size: 1.1rem; flex-shrink: 0; }
.risk-detail { flex: 1; }
.risk-name { font-weight: 600; font-size: 0.9rem; color: var(--text); margin-bottom: 0.15rem; }
.risk-clause { font-size: 0.78rem; color: var(--muted); }

/* ── Summary card ─────────────────────────────────────── */
.summary-container {
    background: linear-gradient(135deg, #f8faff 0%, #eef4ff 100%);
    border: 1px solid #CBD5E1;
    border-left: 4px solid var(--accent);
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    font-size: 0.9rem;
    line-height: 1.8;
    color: var(--text);
    white-space: pre-wrap;
    font-family: 'Inter', sans-serif;
}

/* ── Metric cards override ────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: var(--navy) !important;
    font-family: 'DM Serif Display', serif !important;
}

/* ── Upload area ──────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 0.5rem;
    background: #FAFBFC;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent); }

/* ── Analyze button ───────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.35) !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.45) !important;
}

/* ── Dataframe ────────────────────────────────────────── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Alert override ───────────────────────────────────── */
.stAlert { border-radius: 10px !important; }

/* ── Tabs ─────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 2px solid var(--border) !important;
    background: transparent !important;
}
[data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    padding: 0.5rem 1.25rem !important;
    font-weight: 500 !important;
}

/* ── Footer ───────────────────────────────────────────── */
.footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="logo-mark">⚖️</div>
    <div class="header-text">
        <p class="header-title">AI Contract Risk Analyzer</p>
        <p class="header-subtitle">
            Upload a contract to automatically extract clauses, detect risk provisions,
            and generate an executive summary — powered by NLP.
        </p>
    </div>
    <div class="header-badge">Enterprise Edition</div>
</div>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def extract_text_from_upload(uploaded_file) -> str:
    """Extract plain text from .txt or .pdf uploads."""
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                return "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            except ImportError:
                st.error("PDF parsing library not found. Install pdfplumber: `pip install pdfplumber`")
                return ""
    return ""


def call_api(text: str) -> dict | None:
    """POST to the FastAPI risk-analyze endpoint."""
    try:
        resp = requests.post(
            f"{API_BASE}/risk-analyze",
            json={"text": text, "risk_level": None, "top_n": None},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ **Cannot connect to the backend API.**  \n"
            "Make sure `python3 app.py` is running at `http://127.0.0.1:8080`."
        )
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


def severity_icon(level: str) -> str:
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(level.upper(), "⚪")


def severity_class(level: str) -> str:
    return {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}.get(level.upper(), "low")


def badge_html(level: str) -> str:
    cls = severity_class(level)
    return f'<span class="badge badge-{cls}">{level}</span>'


# ─── Input Section ────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading"><span class="icon">📂</span>Upload Contract</div>', unsafe_allow_html=True)

with st.container():
    col_upload, col_paste = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="input-label">Option 1 — Upload a file</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            label="Upload PDF or TXT",
            type=["pdf", "txt"],
            label_visibility="collapsed",
            help="Supported formats: PDF, TXT",
        )
        if uploaded_file:
            st.success(f"✅ **{uploaded_file.name}** uploaded ({uploaded_file.size:,} bytes)")

    with col_paste:
        st.markdown('<div class="input-label">Option 2 — Paste contract text</div>', unsafe_allow_html=True)
        pasted_text = st.text_area(
            label="Paste contract text",
            label_visibility="collapsed",
            placeholder="Paste the full contract text here…",
            height=180,
        )

# ─── Analyze Button ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([2, 2, 2])
with btn_col:
    analyze_clicked = st.button("🔍 Analyze Contract", use_container_width=True)

# ─── Analysis Logic ───────────────────────────────────────────────────────────
if analyze_clicked:
    # Resolve input — file takes priority
    contract_text = ""
    if uploaded_file:
        uploaded_file.seek(0)
        contract_text = extract_text_from_upload(uploaded_file)
    elif pasted_text.strip():
        contract_text = pasted_text.strip()

    if not contract_text:
        st.warning("⚠️ Please upload a file or paste contract text before analyzing.")
        st.stop()

    with st.spinner("Analyzing contract — extracting clauses, detecting risks, generating summary…"):
        result = call_api(contract_text)

    if result is None:
        st.stop()

    # ── Parse result ──────────────────────────────────────────────────────────
    clauses   = result.get("clauses", [])
    all_risks: list[dict] = []
    for clause in clauses:
        for risk in clause.get("risks", []):
            all_risks.append({
                "risk":         risk.get("risk", ""),
                "level":        risk.get("level", "").upper(),
                "clause_title": clause.get("clause_title", ""),
                "clause_num":   clause.get("clause_number", ""),
            })

    summary = result.get("summary", "No summary generated.")

    high_risks   = [r for r in all_risks if r["level"] == "HIGH"]
    medium_risks = [r for r in all_risks if r["level"] == "MEDIUM"]
    low_risks    = [r for r in all_risks if r["level"] == "LOW"]

    # ── Success banner ────────────────────────────────────────────────────────
    st.success(
        f"✅ Analysis complete — **{len(clauses)} clauses** detected, "
        f"**{len(all_risks)} risks** identified."
    )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # 1. CLAUSE DETECTION
    # ═══════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-heading"><span class="icon">📄</span>Detected Clauses</div>',
        unsafe_allow_html=True,
    )

    if clauses:
        clause_df = pd.DataFrame([
            {
                "Clause No":    c.get("clause_number", ""),
                "Title":        c.get("clause_title", ""),
                "Type":         c.get("clause_type", ""),
            }
            for c in clauses
        ])
        st.dataframe(
            clause_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Clause No": st.column_config.NumberColumn(width="small"),
                "Title":     st.column_config.TextColumn(width="medium"),
                "Type":      st.column_config.TextColumn(width="medium"),
            },
        )
    else:
        st.info("No clauses detected.")

    # ═══════════════════════════════════════════════════════════════
    # 2. RISK ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-heading"><span class="icon">⚠️</span>Risk Analysis</div>',
        unsafe_allow_html=True,
    )

    if all_risks:
        for risk in all_risks:
            lvl   = risk["level"]
            cls   = severity_class(lvl)
            icon  = severity_icon(lvl)
            st.markdown(f"""
            <div class="risk-row {cls}">
                <span class="risk-icon">{icon}</span>
                <div class="risk-detail">
                    <div class="risk-name">{risk['risk']}</div>
                    <div class="risk-clause">Clause {risk['clause_num']}: {risk['clause_title']}</div>
                </div>
                {badge_html(lvl)}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No risks detected.")

    # ═══════════════════════════════════════════════════════════════
    # 3. EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-heading"><span class="icon">📝</span>Executive Summary</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="summary-container">{summary}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # 4. ANALYTICS DASHBOARD
    # ═══════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-heading"><span class="icon">📊</span>Contract Analytics Dashboard</div>',
        unsafe_allow_html=True,
    )

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📋 Total Clauses",  len(clauses))
    m2.metric("🔴 High Risks",    len(high_risks))
    m3.metric("🟡 Medium Risks",  len(medium_risks))
    m4.metric("🟢 Low Risks",     len(low_risks))

    st.markdown("<br>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2, gap="large")

    # ── Chart 1: Clause Distribution ─────────────────────────────
    with chart_col1:
        if clauses:
            type_counts = (
                pd.Series([c.get("clause_type", "Unknown") for c in clauses])
                .value_counts()
                .reset_index()
            )
            type_counts.columns = ["Clause Type", "Count"]

            fig_bar = px.bar(
                type_counts,
                x="Clause Type",
                y="Count",
                title="Clause Distribution",
                color="Count",
                color_continuous_scale=["#93C5FD", "#2A5C99", "#0F1E36"],
                text="Count",
            )
            fig_bar.update_traces(
                textposition="outside",
                marker_line_width=0,
                textfont=dict(size=13, color="#1A202C"),
            )
            fig_bar.update_layout(
                title=dict(font=dict(size=16, color="#0F1E36"), x=0.02),
                paper_bgcolor="white",
                plot_bgcolor="#F7F9FC",
                showlegend=False,
                coloraxis_showscale=False,
                xaxis=dict(
                    title="", tickangle=-30,
                    tickfont=dict(size=11),
                    gridcolor="#E2E8F0",
                ),
                yaxis=dict(
                    title="Count", gridcolor="#E2E8F0",
                    tickfont=dict(size=11),
                ),
                margin=dict(l=20, r=20, t=50, b=80),
                height=380,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No clause data to chart.")

    # ── Chart 2: Risk Severity Distribution ──────────────────────
    with chart_col2:
        if all_risks:
            sev_data = {
                "Severity": ["HIGH", "MEDIUM", "LOW"],
                "Count":    [len(high_risks), len(medium_risks), len(low_risks)],
            }
            sev_df = pd.DataFrame(sev_data)
            sev_df = sev_df[sev_df["Count"] > 0]

            color_map = {"HIGH": "#DC2626", "MEDIUM": "#D97706", "LOW": "#16A34A"}
            colors = [color_map[s] for s in sev_df["Severity"]]

            fig_donut = go.Figure(data=[go.Pie(
                labels=sev_df["Severity"],
                values=sev_df["Count"],
                hole=0.52,
                marker=dict(colors=colors, line=dict(color="white", width=3)),
                textinfo="label+percent",
                textfont=dict(size=13),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            )])

            fig_donut.update_layout(
                title=dict(
                    text="Risk Severity Distribution",
                    font=dict(size=16, color="#0F1E36"),
                    x=0.02,
                ),
                paper_bgcolor="white",
                showlegend=True,
                legend=dict(
                    orientation="v", x=0.85, y=0.5,
                    font=dict(size=12),
                ),
                annotations=[dict(
                    text=f"<b>{len(all_risks)}</b><br>risks",
                    x=0.5, y=0.5,
                    font=dict(size=15, color="#0F1E36"),
                    showarrow=False,
                )],
                margin=dict(l=20, r=100, t=50, b=20),
                height=380,
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No risk data to chart.")

    # ── Report ID ────────────────────────────────────────────────
    if result.get("id"):
        st.markdown(
            f'<div style="text-align:right;font-size:0.78rem;color:#94A3B8;margin-top:0.5rem;">'
            f'Report ID: <code>{result["id"]}</code></div>',
            unsafe_allow_html=True,
        )

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:3.5rem 2rem; background:#fff;
                border:2px dashed #E2E8F0; border-radius:16px; color:#94A3B8;">
        <div style="font-size:3.5rem; margin-bottom:1rem;">⚖️</div>
        <div style="font-size:1.1rem; font-weight:600; color:#64748B; margin-bottom:0.5rem;">
            Ready to analyze your contract
        </div>
        <div style="font-size:0.9rem; max-width:400px; margin:0 auto; line-height:1.6;">
            Upload a PDF or TXT file, or paste contract text above,<br>
            then click <strong>Analyze Contract</strong> to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    AI Contract Risk Analyzer &nbsp;·&nbsp; Powered by FastAPI + Streamlit &nbsp;·&nbsp;
    <span style="color:#3B82F6;">Enterprise Edition</span>
</div>
""", unsafe_allow_html=True)
