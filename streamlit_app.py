"""
AI Contract Risk Analyzer — Streamlit Frontend (Enhanced v2)
Connects to the FastAPI backend at http://127.0.0.1:8080/api/v1
"""

import io
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Contract Risk Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_BASE = "http://127.0.0.1:8080/api/v1"

# ─── Clause description mapping (for chart tooltips) ──────────────────────────
CLAUSE_DESCRIPTIONS = {
    "Termination":                        "Clauses governing termination rights, conditions, and procedures.",
    "Confidentiality":                    "Clauses protecting confidential information and disclosure obligations.",
    "Payment":                            "Clauses defining payment obligations, schedules, and timelines.",
    "Compensation":                       "Clauses covering compensation, fees, and financial arrangements.",
    "Indemnification":                    "Clauses specifying indemnification duties and liability exposure.",
    "Miscellaneous":                      "General miscellaneous provisions and boilerplate terms.",
    "Notices":                            "Clauses governing formal notice requirements and communication.",
    "Investor":                           "Clauses relating to investor rights, obligations, and protections.",
    "Agent Services":                     "Clauses defining agent roles, duties, and service scope.",
    "Expenses And Fees":                  "Clauses covering expense reimbursement, fees, and cost allocation.",
    "Term":                               "Clauses defining the duration, renewal, and expiration of the agreement.",
    "Royalty":                            "Clauses covering royalty payments, schedules, and guarantees.",
    "Definitions":                        "Clauses providing definitions of key terms used in the contract.",
    "Rights And Obligations Upon Termination": "Clauses specifying rights and duties that survive termination.",
    "Independent Contractors":            "Clauses establishing the independent contractor relationship.",
    "Services":                           "Clauses outlining the scope of services to be provided.",
}

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
    --navy:    #0F1E36;
    --slate:   #1C3358;
    --blue:    #2A5C99;
    --accent:  #3B82F6;
    --gold:    #E8B84B;
    --surface: #F1F5F9;
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

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }
.main { background: var(--surface); }
.block-container { padding: 0 2rem 4rem 2rem !important; max-width: 1340px; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, var(--navy) 0%, var(--slate) 60%, #1e4a7a 100%);
    border-radius: 0 0 24px 24px;
    padding: 2.5rem 3rem;
    margin: -1rem -2rem 2.5rem -2rem;
    display: flex; align-items: center; gap: 2rem;
    position: relative; overflow: hidden;
}
.app-header::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.app-header::after {
    content: ''; position: absolute; bottom: -40px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(232,184,75,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.logo-mark {
    width: 68px; height: 68px;
    background: linear-gradient(135deg, var(--gold), #f59e0b);
    border-radius: 16px; display: flex; align-items: center; justify-content: center;
    font-size: 2rem; box-shadow: 0 8px 24px rgba(232,184,75,0.35);
    flex-shrink: 0; position: relative; z-index: 1;
}
.header-text { position: relative; z-index: 1; }
.header-title {
    font-family: 'DM Serif Display', serif; font-size: 2.1rem;
    color: #FFFFFF; margin: 0; line-height: 1.15; letter-spacing: -0.5px;
}
.header-subtitle {
    font-size: 0.95rem; color: rgba(255,255,255,0.65);
    margin: 0.4rem 0 0 0; font-weight: 400; max-width: 560px; line-height: 1.5;
}
.header-badge {
    margin-left: auto; position: relative; z-index: 1;
    background: rgba(59,130,246,0.2); border: 1px solid rgba(59,130,246,0.35);
    color: #93C5FD; font-size: 0.78rem; font-weight: 600;
    padding: 0.35rem 0.85rem; border-radius: 20px; letter-spacing: 0.5px;
    text-transform: uppercase; white-space: nowrap;
}

/* ── Section headings ── */
.section-heading {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 1.15rem; font-weight: 700; color: var(--navy);
    margin: 2rem 0 1rem; padding-bottom: 0.6rem;
    border-bottom: 2px solid var(--border); letter-spacing: -0.2px;
}

/* ── Input labels ── */
.input-label {
    font-size: 0.82rem; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.5rem;
}

/* ── Enterprise metric cards ── */
.kpi-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1.1rem; margin-bottom: 1.75rem;
}
.kpi-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; padding: 1.4rem 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    text-align: center; position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    border-radius: 16px 16px 0 0;
}
.kpi-card.total::before  { background: linear-gradient(90deg, var(--accent), #60a5fa); }
.kpi-card.high::before   { background: linear-gradient(90deg, var(--red), #f87171); }
.kpi-card.medium::before { background: linear-gradient(90deg, var(--amber), #fbbf24); }
.kpi-card.low::before    { background: linear-gradient(90deg, var(--green), #4ade80); }
.kpi-label {
    font-size: 0.73rem; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.6rem;
}
.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem; line-height: 1; margin-bottom: 0.4rem;
}
.kpi-card.total  .kpi-value { color: var(--accent); }
.kpi-card.high   .kpi-value { color: var(--red); }
.kpi-card.medium .kpi-value { color: var(--amber); }
.kpi-card.low    .kpi-value { color: var(--green); }
.kpi-sub {
    font-size: 0.76rem; color: var(--muted); font-weight: 500;
}

/* ── Filter panel ── */
.filter-panel {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.25rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 1.5rem;
}
.filter-title {
    font-size: 0.8rem; font-weight: 700; color: var(--navy);
    text-transform: uppercase; letter-spacing: 0.7px;
    margin-bottom: 0.9rem; display: flex; align-items: center; gap: 0.45rem;
}

/* ── Risk badges ── */
.badge {
    display: inline-block; padding: 0.25rem 0.65rem;
    border-radius: 20px; font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.5px; text-transform: uppercase;
}
.badge-high   { background: var(--red-bg);   color: var(--red);   border: 1px solid #FECACA; }
.badge-medium { background: var(--amber-bg); color: var(--amber); border: 1px solid #FDE68A; }
.badge-low    { background: var(--green-bg); color: var(--green); border: 1px solid #BBF7D0; }

/* ── Risk rows ── */
.risk-row {
    display: flex; align-items: center; gap: 1rem;
    padding: 0.9rem 1.1rem; border-radius: 10px;
    margin-bottom: 0.6rem; border: 1px solid transparent;
}
.risk-row.high   { background: var(--red-bg);   border-color: #FECACA; }
.risk-row.medium { background: var(--amber-bg); border-color: #FDE68A; }
.risk-row.low    { background: var(--green-bg); border-color: #BBF7D0; }
.risk-icon { font-size: 1.1rem; flex-shrink: 0; }
.risk-detail { flex: 1; }
.risk-name  { font-weight: 600; font-size: 0.9rem; color: var(--text); margin-bottom: 0.15rem; }
.risk-clause{ font-size: 0.78rem; color: var(--muted); }

/* ── Summary ── */
.summary-container {
    background: linear-gradient(135deg, #f8faff 0%, #eef4ff 100%);
    border: 1px solid #CBD5E1; border-left: 4px solid var(--accent);
    border-radius: 12px; padding: 1.5rem 1.75rem;
    font-size: 0.9rem; line-height: 1.8; color: var(--text);
    white-space: pre-wrap; font-family: 'Inter', sans-serif;
}

/* ── Upload ── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border); border-radius: 12px;
    padding: 0.5rem; background: #FAFBFC; transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent); }

/* ── Analyze button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #1d4ed8) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.75rem 2.5rem !important;
    font-size: 1rem !important; font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.35) !important;
    transition: all 0.2s ease !important; width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.45) !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
.stAlert     { border-radius: 10px !important; }

/* ── Footer ── */
.footer {
    text-align: center; color: var(--muted); font-size: 0.8rem;
    padding: 2rem 0 1rem; border-top: 1px solid var(--border); margin-top: 3rem;
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
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                st.error("PDF parsing library not found. Install pdfplumber: `pip install pdfplumber`")
                return ""
    return ""


def call_api(text: str) -> dict | None:
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
    return f'<span class="badge badge-{severity_class(level)}">{level}</span>'

def kpi_card(css_class: str, label: str, value: int, sub: str) -> str:
    return f"""
    <div class="kpi-card {css_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


# ─── Input Section ────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-heading"><span>📂</span> Upload Contract</div>',
    unsafe_allow_html=True,
)

col_upload, col_paste = st.columns([1, 1], gap="large")
with col_upload:
    st.markdown('<div class="input-label">Option 1 — Upload a file</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Upload PDF or TXT", type=["pdf", "txt"],
        label_visibility="collapsed", help="Supported formats: PDF, TXT",
    )
    if uploaded_file:
        st.success(f"✅ **{uploaded_file.name}** uploaded ({uploaded_file.size:,} bytes)")

with col_paste:
    st.markdown('<div class="input-label">Option 2 — Paste contract text</div>', unsafe_allow_html=True)
    pasted_text = st.text_area(
        label="Paste contract text", label_visibility="collapsed",
        placeholder="Paste the full contract text here…", height=180,
    )

# ─── Analyze Button ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([2, 2, 2])
with btn_col:
    analyze_clicked = st.button("🔍 Analyze Contract", use_container_width=True)


# ─── Analysis Logic ───────────────────────────────────────────────────────────
if analyze_clicked:
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

    # Persist result so filter widget interactions don't wipe it
    st.session_state["analysis_result"] = result

# ── Render results from session_state (survives filter widget reruns) ──────────
if "analysis_result" in st.session_state:
    result = st.session_state["analysis_result"]

    # ── Parse raw result ──────────────────────────────────────────────────────
    clauses = result.get("clauses", [])
    all_risks: list[dict] = []
    for clause in clauses:
        for risk in clause.get("risks", []):
            all_risks.append({
                "risk":         risk.get("risk", ""),
                "level":        risk.get("level", "").upper(),
                "clause_title": clause.get("clause_title", ""),
                "clause_type":  clause.get("clause_type", ""),
                "clause_num":   clause.get("clause_number", ""),
            })

    summary = result.get("summary", "No summary generated.")

    high_risks   = [r for r in all_risks if r["level"] == "HIGH"]
    medium_risks = [r for r in all_risks if r["level"] == "MEDIUM"]
    low_risks    = [r for r in all_risks if r["level"] == "LOW"]

    # unique clause types for filter
    clause_types_available = sorted({c.get("clause_type", "Unknown") for c in clauses})

    # ── Success banner ────────────────────────────────────────────────────────
    st.success(
        f"✅ Analysis complete — **{len(clauses)} clauses** detected, "
        f"**{len(all_risks)} risks** identified."
    )
    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — ANALYTICS DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-heading"><span>📊</span> Contract Analytics Dashboard</div>',
        unsafe_allow_html=True,
    )

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    cards_html = (
        '<div class="kpi-grid">'
        + kpi_card("total",  "📋 Total Clauses",  len(clauses),      "Detected clause groups")
        + kpi_card("high",   "🔴 High Risks",     len(high_risks),   "Critical provisions")
        + kpi_card("medium", "🟡 Medium Risks",   len(medium_risks), "Notable provisions")
        + kpi_card("low",    "🟢 Low Risks",      len(low_risks),    "Minor provisions")
        + "</div>"
    )
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Charts (equal height) ─────────────────────────────────────────────────
    CHART_HEIGHT = 400

    chart_col1, chart_col2 = st.columns(2, gap="large")

    # Bar chart — Clause Distribution Analysis
    with chart_col1:
        if clauses:
            type_counts = (
                pd.Series([c.get("clause_type", "Unknown") for c in clauses])
                .value_counts()
                .rename_axis("Clause Type")
                .reset_index(name="Count")
                .sort_values(
                    by=["Count", "Clause Type"],
                    ascending=[False, True]
                )
                .reset_index(drop=True)
            )
            type_counts.columns = ["Clause Type", "Count"]
            type_counts["Description"] = type_counts["Clause Type"].map(
                lambda t: CLAUSE_DESCRIPTIONS.get(t, "General contract provision.")
            )

            fig_bar = go.Figure()
            colors_bar = [
                f"rgba(42,92,153,{0.6 + 0.4 * (i / max(len(type_counts) - 1, 1))})"
                for i in range(len(type_counts))
            ]
            fig_bar.add_trace(go.Bar(
                x=type_counts["Clause Type"],
                y=type_counts["Count"],
                text=type_counts["Count"],
                textposition="outside",
                textfont=dict(size=13, color="#1A202C", family="Inter"),
                marker=dict(
                    color=colors_bar,
                    line=dict(width=0),
                    cornerradius=6,
                ),
                customdata=list(zip(type_counts["Count"], type_counts["Description"])),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Count: <b>%{customdata[0]}</b><br><br>"
                    "<i>%{customdata[1]}</i><extra></extra>"
                ),
            ))
            fig_bar.update_layout(
                title=dict(
                    text="<b>Clause Distribution Analysis</b>",
                    font=dict(size=16, color="#0F1E36", family="DM Serif Display"),
                    x=0.5, xanchor="center",
                ),
                paper_bgcolor="white",
                plot_bgcolor="#F7F9FC",
                showlegend=False,
                xaxis=dict(
                    title=dict(text="Clause Categories", font=dict(size=12, color="#64748B")),
                    tickangle=-30, tickfont=dict(size=11, family="Inter"),
                    gridcolor="#E2E8F0", linecolor="#E2E8F0",
                ),
                yaxis=dict(
                    title=dict(text="Number of Clauses", font=dict(size=12, color="#64748B")),
                    tickfont=dict(size=11, family="Inter"),
                    gridcolor="#E2E8F0", linecolor="#E2E8F0",
                    zeroline=False,
                ),
                margin=dict(l=50, r=30, t=60, b=100),
                height=CHART_HEIGHT,
                hoverlabel=dict(
                    bgcolor="white", bordercolor="#E2E8F0",
                    font=dict(size=13, family="Inter"),
                ),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No clause data to chart.")

    # Donut chart — Risk Severity Distribution
    with chart_col2:
        if all_risks:
            risk_by_level = {"HIGH": high_risks, "MEDIUM": medium_risks, "LOW": low_risks}
            sev_labels  = [k for k, v in risk_by_level.items() if v]
            sev_counts  = [len(v) for v in risk_by_level.values() if v]
            sev_colors  = {"HIGH": "#DC2626", "MEDIUM": "#D97706", "LOW": "#16A34A"}
            sev_colors_list = [sev_colors[l] for l in sev_labels]

            # build top-risk text per segment for tooltip
            top_risk_texts = []
            for lvl in sev_labels:
                risks_for_level = risk_by_level[lvl]
                top5 = [r["risk"] for r in risks_for_level[:5]]
                bullet_list = "<br>".join(f"• {r}" for r in top5)
                top_risk_texts.append(bullet_list)

            fig_donut = go.Figure(data=[go.Pie(
                labels=sev_labels,
                values=sev_counts,
                hole=0.55,
                marker=dict(colors=sev_colors_list, line=dict(color="white", width=3)),
                textinfo="label+percent",
                textfont=dict(size=13, family="Inter"),
                customdata=top_risk_texts,
                hovertemplate=(
                    "<b>Risk Level: %{label}</b><br>"
                    "Count: <b>%{value}</b><br>"
                    "Share: %{percent}<br><br>"
                    "<b>Top Risks:</b><br>%{customdata}<extra></extra>"
                ),
                pull=[0.04 if l == "HIGH" else 0 for l in sev_labels],
            )])
            fig_donut.update_layout(
                title=dict(
                    text="<b>Risk Severity Distribution</b>",
                    font=dict(size=16, color="#0F1E36", family="DM Serif Display"),
                    x=0.5, xanchor="center",
                ),
                paper_bgcolor="white",
                showlegend=True,
                legend=dict(
                    orientation="v", x=0.82, y=0.5,
                    font=dict(size=12, family="Inter"),
                    itemsizing="constant",
                ),
                annotations=[dict(
                    text=f"<b style='font-size:22px'>{len(all_risks)}</b><br>"
                         "<span style='font-size:11px;color:#64748B'>total risks</span>",
                    x=0.5, y=0.5, font=dict(size=13, color="#0F1E36"),
                    showarrow=False,
                )],
                margin=dict(l=20, r=110, t=60, b=30),
                height=CHART_HEIGHT,
                hoverlabel=dict(
                    bgcolor="white", bordercolor="#E2E8F0",
                    font=dict(size=13, family="Inter"),
                    align="left",
                ),
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No risk data to chart.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — FILTER PANEL
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-heading"><span>🎛️</span>  Analysis Filters </div>',
        unsafe_allow_html=True,
    )

    with st.container():
        f1, f2, f3 = st.columns([1.2, 1, 1.6], gap="large")

        with f1:
            severity_filter = st.selectbox(
                "Risk Severity",
                options=["All Risks", "High Risk Only", "Medium Risk Only", "Low Risk Only"],
                index=0,
            )

        with f2:
            top_n_filter = st.number_input(
                "Number of Top Risks to Display",
                min_value=1,
                max_value=max(len(all_risks), 1),
                value=min(5, max(len(all_risks), 1)),
                step=1,
            )

        with f3:
            clause_type_filter = st.selectbox(
                "Clause Type",
                options=["All Clauses"] + clause_types_available,
                index=0,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Apply filters ─────────────────────────────────────────────────────────
    level_map = {
        "All Risks":       None,
        "High Risk Only":  "HIGH",
        "Medium Risk Only":"MEDIUM",
        "Low Risk Only":   "LOW",
    }
    chosen_level = level_map[severity_filter]

    filtered_risks = all_risks[:]
    if chosen_level:
        filtered_risks = [r for r in filtered_risks if r["level"] == chosen_level]
    filtered_risks = filtered_risks[:int(top_n_filter)]

    filtered_clauses = clauses[:]
    if clause_type_filter != "All Clauses":
        filtered_clauses = [c for c in filtered_clauses if c.get("clause_type") == clause_type_filter]

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — CLAUSE DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-heading"><span>📄</span> Detected Clauses</div>',
        unsafe_allow_html=True,
    )

    if filtered_clauses:
        clause_df = pd.DataFrame([
            {
                "Clause No": c.get("clause_number", ""),
                "Title":     c.get("clause_title", ""),
                "Type":      c.get("clause_type", ""),
            }
            for c in filtered_clauses
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
        st.info("No clauses match the selected filter.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — RISK ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-heading"><span>⚠️</span> Risk Analysis</div>',
        unsafe_allow_html=True,
    )

    if filtered_risks:
        for risk in filtered_risks:
            lvl  = risk["level"]
            cls  = severity_class(lvl)
            icon = severity_icon(lvl)
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
        st.info("No risks match the selected filter.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-heading"><span>📝</span> Executive Summary</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="summary-container">{summary}</div>', unsafe_allow_html=True)

    # ── Report ID ──────────────────────────────────────────────────────────────
    if result.get("id"):
        st.markdown(
            f'<div style="text-align:right;font-size:0.78rem;color:#94A3B8;margin-top:0.75rem;">'
            f'Report ID: <code>{result["id"]}</code></div>',
            unsafe_allow_html=True,
        )

else:
    # ── Empty state (no analysis run yet) ────────────────────────────────────
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