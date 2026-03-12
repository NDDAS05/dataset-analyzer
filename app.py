"""
app.py — Streamlit Web Frontend for the Automated Dataset Analyzer
===================================================================
Run with:
    streamlit run app.py
"""

import sys
import os
import io
import base64
import tempfile

import streamlit as st
import pandas as pd

# ── Point to src/ ─────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(ROOT_DIR, 'src')
sys.path.insert(0, SRC_DIR)

from analyzer         import analyze_dataset, detect_target_column, generate_insights
from visualizer       import (generate_visualizations, plot_feature_importance,
                               plot_confusion_matrix)
from ml_insights      import run_ml_analysis
from report_generator import generate_report

# ── Cached heavy functions ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_load(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load and cache DataFrame — only re-runs when file actually changes."""
    import io as _io
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            df = pd.read_csv(_io.BytesIO(file_bytes), encoding=enc, on_bad_lines='skip')
            if not df.empty and len(df.columns) >= 2:
                return df.dropna(how='all')
        except Exception:
            continue
    raise ValueError("Could not parse the file. Please check it is a valid CSV.")

@st.cache_data(show_spinner=False)
def cached_analyze(file_bytes: bytes) -> dict:
    """Cache EDA analysis — only re-runs when file changes."""
    df = cached_load(file_bytes, "")
    return analyze_dataset(df)

@st.cache_data(show_spinner=False)
def cached_ml(file_bytes: bytes, target: str) -> dict:
    """Cache ML result — only re-runs when file or target changes."""
    df = cached_load(file_bytes, "")
    return run_ml_analysis(df, target)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Dataset Analyzer · Nirupam Das",
    page_icon   = "🔬",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Nunito+Sans:wght@400;500;600;700&display=swap');

/* ── Base & fonts ─────────────────────────────────────────── */
html, body, [class*="css"],
[data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"],
p, span, div, label, h1, h2, h3 {
    font-family: 'Nunito Sans', sans-serif !important;
    color: #2D3250;
}

/* ── App background ───────────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {
    background-color: #F4F6FF !important;
}

/* ── Hide Streamlit chrome ────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"]  { display: none; }
[data-testid="stDecoration"]{ display: none; }

/* ── Sidebar — no horizontal scroll ──────────────────────── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1.5px solid #DDE3F7 !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
}
[data-testid="stSidebar"] > div:first-child {
    overflow-x: hidden !important;
    overflow-y: auto !important;
    max-width: 100% !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Nunito Sans', sans-serif !important;
    max-width: 100% !important;
    word-break: break-word !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
}
/* Sidebar text inputs */
[data-testid="stSidebar"] input {
    background: #F4F6FF !important;
    color: #2D3250 !important;
    border: 1.5px solid #DDE3F7 !important;
    border-radius: 8px !important;
}
/* Sidebar labels and markdown text */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #2D3250 !important;
}
[data-testid="stSidebar"] .stMarkdown p { color: #525B84 !important; }

/* ── Streamlit native widgets — force readable text ───────── */
/* All markdown text */
.stMarkdown p, .stMarkdown span, .stMarkdown li { color: #2D3250 !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3  { color: #2D3250 !important; }

/* st.metric labels & values */
[data-testid="stMetric"] label { color: #8A93B2 !important; font-size: 12px !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #2D3250 !important; font-weight: 800 !important; }

/* st.tabs */
[data-testid="stTabs"] [role="tab"] {
    color: #525B84 !important;
    font-weight: 700 !important;
    font-size: 13px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #6C8EF5 !important;
    border-bottom-color: #6C8EF5 !important;
}
[data-testid="stTabs"] { overflow-x: auto !important; }

/* st.selectbox */
[data-testid="stSelectbox"] label { color: #2D3250 !important; font-weight: 700 !important; }
[data-testid="stSelectbox"] > div > div {
    background: #FFFFFF !important;
    border: 1.5px solid #DDE3F7 !important;
    border-radius: 10px !important;
    color: #2D3250 !important;
}

/* st.checkbox */
[data-testid="stCheckbox"] span { color: #2D3250 !important; font-weight: 600 !important; }

/* st.text_input */
[data-testid="stTextInput"] label { color: #2D3250 !important; font-weight: 700 !important; }
[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    color: #2D3250 !important;
    border: 1.5px solid #DDE3F7 !important;
    border-radius: 10px !important;
}

/* st.file_uploader */
[data-testid="stFileUploader"] label { color: #2D3250 !important; }
[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF !important;
    border: 2px dashed #C4D4FC !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span { color: #8A93B2 !important; }

/* st.dataframe */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }

/* st.spinner */
[data-testid="stSpinner"] p { color: #6C8EF5 !important; }

/* Download button */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #6C8EF5, #B97FF5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    font-size: 15px !important;
    padding: 14px 32px !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(108,142,245,0.35) !important;
}

/* st.success / st.info / st.error */
[data-testid="stAlert"] p { color: #2D3250 !important; }

/* ── Custom component classes ─────────────────────────────── */
.stat-card {
    background: #FFFFFF;
    border: 1.5px solid #DDE3F7;
    border-radius: 16px;
    padding: 20px 18px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(108,142,245,0.10);
    margin-bottom: 8px;
}
.stat-label {
    font-size: 11px !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: #8A93B2 !important; margin-bottom: 6px;
}
.stat-value {
    font-size: 28px !important; font-weight: 900 !important;
    line-height: 1; margin-bottom: 4px;
}
.stat-sub { font-size: 11px !important; color: #8A93B2 !important; font-weight: 500 !important; }

.blue-val   { color: #6C8EF5 !important; }
.coral-val  { color: #F4845F !important; }
.green-val  { color: #52C17E !important; }
.purple-val { color: #B97FF5 !important; }

.section-title {
    font-size: 20px; font-weight: 800;
    color: #2D3250 !important; margin-bottom: 4px;
}
.section-sub {
    font-size: 12px; color: #8A93B2 !important;
    font-weight: 500; margin-bottom: 20px;
}

.insight-box {
    background: #FFFFFF;
    border: 1.5px solid #DDE3F7;
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex; align-items: flex-start; gap: 14px;
    box-shadow: 0 2px 8px rgba(108,142,245,0.08);
    font-size: 14px;
}
.insight-box, .insight-box * { color: #2D3250 !important; }
.insight-num {
    background: linear-gradient(135deg, #6C8EF5, #B97FF5);
    color: white !important; border-radius: 8px;
    width: 26px; height: 26px; min-width: 26px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 900;
    flex-shrink: 0;
}

.upload-hero {
    background: #FFFFFF;
    border: 2px dashed #C4D4FC;
    border-radius: 20px;
    padding: 48px 32px;
    text-align: center;
    margin: 32px 0;
}
.upload-hero h2 { font-size: 22px; font-weight: 800; color: #2D3250 !important; margin-bottom: 8px; }
.upload-hero p  { color: #8A93B2 !important; font-size: 14px; }

.pill-num {
    background: #EEF2FF; color: #6C8EF5 !important;
    border: 1.5px solid #C4D4FC; border-radius: 20px;
    padding: 4px 12px; font-size: 12px;
    font-weight: 700; display: inline-block; margin: 3px;
}
.pill-cat {
    background: #F6EEFF; color: #B97FF5 !important;
    border: 1.5px solid #DEC8FA; border-radius: 20px;
    padding: 4px 12px; font-size: 12px;
    font-weight: 700; display: inline-block; margin: 3px;
}

/* ── Mobile responsiveness ────────────────────────────────── */
@media (max-width: 768px) {
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="block-container"] { padding: 1rem !important; }
    .stat-card { padding: 14px 12px; }
    .stat-value { font-size: 22px !important; }
    .upload-hero { padding: 28px 16px; }
    .upload-hero h2 { font-size: 18px; }
    [data-testid="stTabs"] [role="tab"] { font-size: 11px !important; padding: 6px 8px !important; }
    .insight-box { padding: 12px 14px; font-size: 13px; }
}
@media (max-width: 480px) {
    [data-testid="block-container"] { padding: 0.5rem !important; }
    .stat-value { font-size: 18px !important; }
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_from_upload(uploaded_file) -> pd.DataFrame:
    """Load DataFrame from a Streamlit UploadedFile, with encoding fallback."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc, on_bad_lines='skip')
            if not df.empty and len(df.columns) >= 2:
                return df.dropna(how='all')
        except Exception:
            continue
    raise ValueError("Could not parse the uploaded file. Please check it is a valid CSV.")


def fig_to_b64(fig) -> str:
    """Convert a matplotlib figure to a base64 PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor=fig.get_facecolor(), dpi=140)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def show_img(b64: str, caption: str = ""):
    """Render a base64 image in Streamlit."""
    if b64:
        st.markdown(
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;border-radius:12px;" alt="{caption}">',
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 20px;">
      <div style="background:linear-gradient(135deg,#6C8EF5,#B97FF5);
                  width:44px;height:44px;border-radius:14px;
                  display:flex;align-items:center;justify-content:center;
                  font-size:22px;margin-bottom:10px;
                  box-shadow:0 4px 12px rgba(108,142,245,0.30);">🔬</div>
      <div style="font-size:16px;font-weight:900;color:#2D3250;">Dataset Analyzer</div>
      <div style="font-size:11px;color:#8A93B2;font-weight:600;">by Nirupam Das</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚙️ Options**")

    run_ml = st.checkbox("Run ML Analysis", value=True)

    target_col_input = st.text_input(
        "ML Target Column",
        placeholder="Auto-detected if blank",
        help="Leave blank to auto-detect. Must match a column name exactly."
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:#8A93B2;line-height:1.7;">
    <b>What this tool does:</b><br>
    📊 Dataset overview<br>
    🔍 Data quality checks<br>
    📈 Feature distributions<br>
    🔗 Correlation analysis<br>
    🏷️ Categorical analysis<br>
    🤖 ML insights (optional)<br>
    💡 Auto-generated insights<br>
    📄 Downloadable HTML report
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:10px;color:#8A93B2;">Upload a CSV to get started</div>',
        unsafe_allow_html=True
    )


# ═════════════════════════════════════════════════════════════════════════════
# Landing / Upload
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="margin-bottom:8px;">
  <span style="background:#EEF2FF;color:#6C8EF5;font-size:11px;font-weight:700;
               letter-spacing:.08em;text-transform:uppercase;padding:5px 12px;
               border-radius:20px;">🔬 Automated EDA</span>
</div>
<h1 style="font-size:32px;font-weight:900;color:#2D3250;margin-bottom:6px;
           font-family:'Nunito',sans-serif;">
  Automated Dataset Analyzer
</h1>
<p style="color:#8A93B2;font-size:14px;font-weight:500;margin-bottom:28px;">
  Upload any CSV — get a full EDA report with charts, quality checks, and ML insights.
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Drop your CSV here",
    type=["csv"],
    label_visibility="collapsed",
)

# ── If nothing uploaded, show hero ───────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div class="upload-hero">
      <div style="font-size:48px;margin-bottom:16px;">📂</div>
      <h2>Drop your CSV file above</h2>
      <p>Supports any CSV — mixed types, missing values, large files.<br>
         The full report will appear instantly on this page.</p>
      <div style="margin-top:24px;display:flex;justify-content:center;gap:12px;flex-wrap:wrap;">
        <span style="background:#EEF2FF;color:#6C8EF5;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:700;">pandas + numpy</span>
        <span style="background:#EDFAF3;color:#52C17E;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:700;">scikit-learn</span>
        <span style="background:#F6EEFF;color:#B97FF5;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:700;">matplotlib + seaborn</span>
        <span style="background:#FFF0EB;color:#F4845F;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:700;">Jinja2 reports</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# Main analysis (runs when file is uploaded)
# ═════════════════════════════════════════════════════════════════════════════

# ── Load data & analysis (cached) ────────────────────────────────────────────
file_bytes = uploaded_file.getvalue()
dataset_name = os.path.splitext(uploaded_file.name)[0].replace('_',' ').replace('-',' ').title()

with st.spinner("Loading and analysing dataset..."):
    try:
        df       = cached_load(file_bytes, uploaded_file.name)
        analysis = cached_analyze(file_bytes)
    except Exception as e:
        st.error(f"❌ Could not load file: {e}")
        st.stop()

ov = analysis['overview']

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">Total Rows</div>
      <div class="stat-value blue-val">{ov['rows']:,}</div>
      <div class="stat-sub">records</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">Columns</div>
      <div class="stat-value purple-val">{ov['columns']}</div>
      <div class="stat-sub">{len(ov['numeric_columns'])} numeric · {len(ov['categorical_columns'])} categ.</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">Missing Data</div>
      <div class="stat-value coral-val">{analysis['total_missing_pct']}%</div>
      <div class="stat-sub">{analysis['total_missing']:,} cells</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">Memory</div>
      <div class="stat-value green-val">{ov['memory_usage_mb']}</div>
      <div class="stat-sub">megabytes</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🗂️ Overview",
    "🔍 Data Quality",
    "📈 Distributions",
    "🔗 Correlations",
    "🏷️ Categorical",
    "🤖 ML Insights",
    "💡 Insights",
    "📄 Download Report",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

    # Column pills
    if ov['numeric_columns']:
        pills = " ".join(f'<span class="pill-num">{c}</span>' for c in ov['numeric_columns'])
        st.markdown(f"<div style='margin-bottom:4px;font-size:11px;font-weight:700;color:#8A93B2;text-transform:uppercase;letter-spacing:.08em;'>Numeric</div>{pills}<br><br>", unsafe_allow_html=True)

    if ov['categorical_columns']:
        pills = " ".join(f'<span class="pill-cat">{c}</span>' for c in ov['categorical_columns'])
        st.markdown(f"<div style='margin-bottom:4px;font-size:11px;font-weight:700;color:#8A93B2;text-transform:uppercase;letter-spacing:.08em;'>Categorical</div>{pills}<br><br>", unsafe_allow_html=True)

    # Descriptive stats
    if ov['numeric_columns']:
        st.markdown("**Descriptive Statistics**")
        st.dataframe(
            df[ov['numeric_columns']].describe().round(3).T,
        )

    st.markdown("**Data Preview**")
    st.dataframe(df.head(20))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Data Quality
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-title">Data Quality</div>', unsafe_allow_html=True)

    q1, q2, q3 = st.columns(3)
    mv  = analysis['missing_values']
    dup = analysis['duplicates']
    out = {c: v for c, v in analysis['outliers'].items() if v['count'] > 0}

    q1.metric("Missing Values",  f"{analysis['total_missing_pct']}%",
              f"{analysis['total_missing']:,} cells")
    q2.metric("Duplicate Rows",  f"{dup['count']:,}", f"{dup['pct']}%")
    q3.metric("Cols with Outliers", str(len(out)))

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Missing Values by Column**")
        if mv:
            mv_df = pd.DataFrame([
                {'Column': c, 'Missing': v['missing_count'], 'Pct': f"{v['missing_pct']}%",
                 'Severity': 'High' if v['missing_pct']>50 else 'Medium' if v['missing_pct']>20 else 'Low'}
                for c, v in mv.items()
            ])
            st.dataframe(mv_df, hide_index=True)
        else:
            st.success("✅ No missing values!")

    with col_b:
        st.markdown("**Outliers by Column (IQR)**")
        if out:
            out_df = pd.DataFrame([
                {'Column': c, 'Count': v['count'], 'Pct': f"{v['pct']}%",
                 'Lower': v['lower_bound'], 'Upper': v['upper_bound']}
                for c, v in out.items()
            ])
            st.dataframe(out_df, hide_index=True)
        else:
            st.success("✅ No outliers detected!")

    # Charts
    with tempfile.TemporaryDirectory() as tmpdir:
        with st.spinner("Generating quality charts..."):
            import matplotlib
            matplotlib.use('Agg')
            from visualizer import _plot_missing_values, _plot_outliers
            mv_path  = _plot_missing_values(analysis, tmpdir)
            out_path = _plot_outliers(analysis, tmpdir)

        if mv_path or out_path:
            ca, cb = st.columns(2)
            if mv_path:
                with ca:
                    st.image(os.path.join(tmpdir, mv_path.split('/')[-1]))
            if out_path:
                with cb:
                    st.image(os.path.join(tmpdir, out_path.split('/')[-1]))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Distributions
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-title">Feature Distributions</div>', unsafe_allow_html=True)
    num_cols = ov['numeric_columns']

    if not num_cols:
        st.info("No numeric columns found.")
    else:
        selected_dist = st.selectbox(
            "Select a feature to inspect:",
            options=num_cols,
            key="dist_select"
        )

        with st.spinner(f"Plotting {selected_dist}..."):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np

            PALETTE = ["#6C8EF5","#F4845F","#52C17E","#B97FF5","#F5C842",
                       "#5BC4F5","#F5A3C0","#7ED8C8","#F5D07A","#A8D8A8"]
            color = PALETTE[num_cols.index(selected_dist) % len(PALETTE)]

            data = df[selected_dist].dropna()
            fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='#FAFBFF')
            ax.set_facecolor('#FFFFFF')
            ax.hist(data, bins=35, color=color, alpha=0.80,
                    edgecolor='white', linewidth=1.2, zorder=3)
            ax.axvline(data.mean(),   color='#F4845F', linestyle='--',
                       linewidth=2.2, label=f'Mean: {data.mean():.2f}', zorder=4)
            ax.axvline(data.median(), color='#52C17E', linestyle=':',
                       linewidth=2.2, label=f'Median: {data.median():.2f}', zorder=4)
            ax.set_title(f'Distribution of  {selected_dist}',
                         color='#2D3250', fontsize=14, fontweight='bold', pad=14)
            ax.set_xlabel(selected_dist, color='#8A93B2')
            ax.set_ylabel('Count',      color='#8A93B2')
            ax.legend(fontsize=10, facecolor='white', edgecolor='#E8ECF8', labelcolor='#2D3250')
            for sp in ['top','right']: ax.spines[sp].set_visible(False)
            for sp in ['left','bottom']: ax.spines[sp].set_color('#E8ECF8')
            ax.grid(True, color='#E8ECF8', linewidth=0.8, zorder=0)
            ax.tick_params(colors='#8A93B2')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Quick stats row
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Mean",   f"{data.mean():.2f}")
        s2.metric("Median", f"{data.median():.2f}")
        s3.metric("Std Dev",f"{data.std():.2f}")
        s4.metric("Min",    f"{data.min():.2f}")
        s5.metric("Max",    f"{data.max():.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Correlations
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-title">Correlation Analysis</div>', unsafe_allow_html=True)
    num_cols = ov['numeric_columns']

    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns for correlation analysis.")
    else:
        with st.spinner("Generating heatmap..."):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np

            corr = df[num_cols].corr()
            n    = len(num_cols)
            size = max(7, min(n * 1.0, 16))
            fig, ax = plt.subplots(figsize=(size, size * 0.85), facecolor='#FAFBFF')
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cmap = sns.diverging_palette(230, 20, s=65, l=62, as_cmap=True)
            sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
                        annot=n<=15, fmt='.2f', square=True, ax=ax,
                        linewidths=2.5, linecolor='#FAFBFF',
                        cbar_kws={'shrink':0.7},
                        annot_kws={'size':9,'color':'#2D3250','weight':'bold'})
            ax.set_title('Correlation Heatmap', color='#2D3250',
                         fontsize=14, fontweight='bold', pad=14)
            ax.tick_params(colors='#8A93B2', labelsize=10)
            ax.set_facecolor('#FFFFFF')
            fig.patch.set_facecolor('#FAFBFF')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        strong = analysis.get('strong_correlations', [])
        if strong:
            st.markdown("**Strong correlations (|r| ≥ 0.70)**")
            sc_df = pd.DataFrame(strong, columns=['Column A', 'Column B', 'Correlation'])
            st.dataframe(sc_df, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Categorical
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-title">Categorical Analysis</div>', unsafe_allow_html=True)
    cat_cols = ov['categorical_columns']

    if not cat_cols:
        st.info("No categorical columns found.")
    else:
        selected_cat = st.selectbox(
            "Select a feature to inspect:",
            options=cat_cols,
            key="cat_select"
        )

        PALETTE = ["#6C8EF5","#F4845F","#52C17E","#B97FF5","#F5C842",
                   "#5BC4F5","#F5A3C0","#7ED8C8","#F5D07A","#A8D8A8"]

        with st.spinner(f"Plotting {selected_cat}..."):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            vc    = df[selected_cat].value_counts().head(10)
            color = PALETTE[cat_cols.index(selected_cat) % len(PALETTE)]

            def lighten(h, a):
                h = h.lstrip('#')
                r,g,b = [int(h[i:i+2],16) for i in (0,2,4)]
                return '#{:02x}{:02x}{:02x}'.format(
                    int(r+(255-r)*a), int(g+(255-g)*a), int(b+(255-b)*a))

            n_bars = len(vc)
            colors = [color if j%2==0 else lighten(color,0.40) for j in range(n_bars)]
            fig, ax = plt.subplots(figsize=(10, max(4, n_bars*0.60+1.8)),
                                   facecolor='#FAFBFF')
            ax.set_facecolor('#FFFFFF')
            bars = ax.barh(vc.index[::-1].astype(str), vc.values[::-1],
                           color=colors[::-1], edgecolor='white',
                           linewidth=1.0, height=0.62, zorder=3)
            for bar, val in zip(bars, vc.values[::-1]):
                ax.text(bar.get_width() + vc.max()*0.013,
                        bar.get_y()+bar.get_height()/2,
                        f'{val:,}', va='center', color='#8A93B2', fontsize=10)
            ax.set_title(f'Category Counts — {selected_cat}',
                         color='#2D3250', fontsize=14, fontweight='bold', pad=14)
            ax.set_xlabel('Count', color='#8A93B2')
            ax.set_xlim(0, vc.max() * 1.20)
            for sp in ['top','right']: ax.spines[sp].set_visible(False)
            for sp in ['left','bottom']: ax.spines[sp].set_color('#E8ECF8')
            ax.grid(True, color='#E8ECF8', linewidth=0.8, zorder=0)
            ax.tick_params(colors='#8A93B2')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Unique count
        st.caption(f"Unique values in **{selected_cat}**: {df[selected_cat].nunique()}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — ML Insights
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-title">Machine Learning Insights</div>', unsafe_allow_html=True)

    if not run_ml:
        st.info("ML analysis is disabled. Enable it in the sidebar.")
    else:
        # Determine target
        target = (target_col_input.strip()
                  if target_col_input.strip()
                  else detect_target_column(df))

        if target not in df.columns:
            st.error(f"Column **{target}** not found. Check the target column name in the sidebar.")
        else:
            st.caption(f"Target column: **{target}**")
            if st.button("🤖 Run ML Analysis", key="run_ml_btn", use_container_width=False):
                st.session_state["ml_result"] = cached_ml(file_bytes, target)
                st.session_state["ml_target"] = target

            ml = st.session_state.get("ml_result")
            if ml is None:
                st.info("Click **Run ML Analysis** above to train a Random Forest on this dataset.")
                st.stop()

            if not ml.get('success'):
                st.error(f"ML failed: {ml.get('error', 'Unknown error')}")
            else:
                task = ml['task']
                st.success(f"Task detected: **{task.title()}**")

                # Metrics
                if task == 'classification':
                    acc = ml['metrics']['accuracy']
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Accuracy",       f"{acc*100:.1f}%")
                    m2.metric("Train Samples",  f"{ml['train_samples']:,}")
                    m3.metric("Test Samples",   f"{ml['test_samples']:,}")
                else:
                    r2, mae, rmse = ml['metrics']['r2'], ml['metrics']['mae'], ml['metrics']['rmse']
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("R² Score",       str(r2))
                    m2.metric("MAE",            str(mae))
                    m3.metric("RMSE",           str(rmse))
                    m4.metric("Train Samples",  f"{ml['train_samples']:,}")

                st.markdown("---")

                col_fi, col_cm = st.columns(2)

                with col_fi:
                    st.markdown("**Feature Importance**")
                    if ml.get('feature_importances'):
                        fi_df = pd.DataFrame(ml['feature_importances'])
                        st.dataframe(fi_df, hide_index=True)

                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        fi   = ml['feature_importances'][:15]
                        feats = [x['feature'] for x in fi]
                        imps  = [x['importance'] for x in fi]
                        PALETTE = ["#6C8EF5","#F4845F","#52C17E","#B97FF5","#F5C842",
                                   "#5BC4F5","#F5A3C0","#7ED8C8","#F5D07A","#A8D8A8"]
                        colors = [PALETTE[i%len(PALETTE)] for i in range(len(feats))]
                        fig, ax = plt.subplots(
                            figsize=(8, max(4, len(feats)*0.55+1.5)), facecolor='#FAFBFF')
                        ax.set_facecolor('#FFFFFF')
                        ax.barh(feats[::-1], imps[::-1], color=colors[::-1],
                                edgecolor='white', linewidth=1.0, height=0.6, zorder=3)
                        for i, val in enumerate(imps[::-1]):
                            ax.text(val+0.001, i, f'{val:.4f}', va='center',
                                    color='#8A93B2', fontsize=9)
                        ax.set_title('Feature Importance', color='#2D3250',
                                     fontsize=13, fontweight='bold', pad=12)
                        for sp in ['top','right']: ax.spines[sp].set_visible(False)
                        ax.grid(True, color='#E8ECF8', linewidth=0.8, zorder=0)
                        ax.tick_params(colors='#8A93B2')
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                with col_cm:
                    if task == 'classification' and ml.get('confusion_matrix'):
                        st.markdown("**Confusion Matrix**")
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        import numpy as np

                        cm     = np.array(ml['confusion_matrix'])
                        labels = ml.get('confusion_matrix_labels', [])
                        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='#FAFBFF')
                        cmap = sns.light_palette('#6C8EF5', as_cmap=True)
                        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                                    xticklabels=labels, yticklabels=labels,
                                    ax=ax, linewidths=2.5, linecolor='#FAFBFF',
                                    annot_kws={'size':12,'weight':'bold','color':'#2D3250'})
                        ax.set_xlabel('Predicted', color='#8A93B2')
                        ax.set_ylabel('Actual',    color='#8A93B2')
                        ax.set_title('Confusion Matrix', color='#2D3250',
                                     fontsize=13, fontweight='bold', pad=12)
                        ax.tick_params(colors='#8A93B2')
                        ax.set_facecolor('#FFFFFF')
                        fig.patch.set_facecolor('#FAFBFF')
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — Insights
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="section-title">Insights</div>', unsafe_allow_html=True)

    ml_for_insights = {'skipped': True}
    if run_ml and target_col_input.strip() in df.columns:
        pass  # would re-run; skip for now to avoid double computation

    insights = generate_insights(analysis, ml_for_insights)

    for i, insight in enumerate(insights, 1):
        st.markdown(f"""
        <div class="insight-box">
          <div class="insight-num">{i}</div>
          <div>{insight}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — Download Report
# ─────────────────────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown('<div class="section-title">Download Full HTML Report</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">A self-contained HTML file with all charts embedded — '
        'open it in any browser, share it, or attach it to an email.</div>',
        unsafe_allow_html=True
    )

    if st.button("⚙️ Generate Report", key="gen_report_btn", use_container_width=False):
        st.session_state["report_bytes"]    = None
        st.session_state["report_filename"] = None
        with st.spinner("Building report — this may take a moment..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                # 1. Ensure _assets dir exists before anything writes there
                assets_dir = os.path.join(tmpdir, '_assets')
                os.makedirs(assets_dir, exist_ok=True)

                # 2. Generate all EDA charts (writes into _assets/)
                chart_paths_report = generate_visualizations(df, analysis, tmpdir)

                # 3. Run ML and generate ML-specific charts
                ml_results = {'skipped': True}
                if run_ml:
                    target = (target_col_input.strip()
                              if target_col_input.strip()
                              else detect_target_column(df))
                    if target in df.columns:
                        ml_results = cached_ml(file_bytes, target)
                        if ml_results.get('success'):
                            if ml_results.get('feature_importances'):
                                fi = plot_feature_importance(
                                    ml_results['feature_importances'],
                                    assets_dir)
                                if fi:
                                    chart_paths_report['feature_importance'] = fi
                            if ml_results.get('confusion_matrix'):
                                cm = plot_confusion_matrix(
                                    ml_results['confusion_matrix'],
                                    ml_results.get('confusion_matrix_labels', []),
                                    assets_dir)
                                if cm:
                                    chart_paths_report['confusion_matrix'] = cm

                # 4. Generate insights
                insights_report = generate_insights(analysis, ml_results)

                # 5. Render HTML report
                report_path = generate_report(
                    df           = df,
                    analysis     = analysis,
                    chart_paths  = chart_paths_report,
                    ml_results   = ml_results,
                    insights     = insights_report,
                    output_dir   = tmpdir,
                    dataset_name = dataset_name,
                )

                with open(report_path, 'rb') as f:
                    report_bytes = f.read()

        st.session_state["report_bytes"]    = report_bytes
        st.session_state["report_filename"] = f"{dataset_name.replace(' ','_')}_report.html"

    if st.session_state.get("report_bytes"):
        st.download_button(
            label     = "⬇️  Download HTML Report",
            data      = st.session_state["report_bytes"],
            file_name = st.session_state["report_filename"],
            mime      = "text/html",
        )
    else:
        st.info("Click **Generate Report** above to build your downloadable report.")

    st.markdown("""
    <div style="background:white;border:1.5px solid #DDE3F7;border-radius:14px;
                padding:20px 24px;margin-top:16px;font-size:13px;color:#525B84;">
    <b>📄 What's in the report?</b><br><br>
    The downloaded file is a single self-contained HTML — all charts are embedded as images.<br>
    No internet connection needed to view it. Open it in Chrome, Firefox, or Safari.
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px;padding-top:20px;border-top:1.5px solid #DDE3F7;
            text-align:center;font-size:11px;color:#8A93B2;font-weight:500;">
  Built by <strong style="color:#6C8EF5;">Nirupam Das</strong> · 
  Automated Dataset Analyzer
</div>
""", unsafe_allow_html=True)
