"""
app.py — Flood Prediction AI for the Ganga-Brahmaputra Basin
IIIT Lucknow StreamFlow Prediction Challenge 2026

Sections:
  🏠 Home        — Hero + key metrics
  🔮 Predict     — CSV upload → live inference
  📊 Explorer    — Interactive charts, feature importance, station KGE
  🧠 How It Works — Pipeline explainer with animations
"""

import os
import io
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flood Prediction AI | Ganga-Brahmaputra Basin",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-primary:   #07101f;
  --bg-card:      #0f1d32;
  --bg-card2:     #162540;
  --accent-blue:  #3b82f6;
  --accent-cyan:  #06b6d4;
  --accent-indigo:#6366f1;
  --accent-teal:  #14b8a6;
  --accent-amber: #f59e0b;
  --accent-red:   #ef4444;
  --accent-green: #22c55e;
  --text-primary: #f1f5f9;
  --text-muted:   #94a3b8;
  --border:       #1e3a5f;
}

html, body, .stApp {
  background-color: var(--bg-primary) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--text-primary) !important;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #060f1e 0%, #0f1d32 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; max-width: 1400px !important; }

/* ── Metric Cards */
.metric-card {
  background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.4rem 1.2rem;
  text-align: center;
  transition: transform 0.25s, box-shadow 0.25s;
  position: relative;
  overflow: hidden;
}
.metric-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  border-radius: 16px 16px 0 0;
}
.metric-card.blue::before   { background: linear-gradient(90deg, #3b82f6, #06b6d4); }
.metric-card.green::before  { background: linear-gradient(90deg, #22c55e, #14b8a6); }
.metric-card.amber::before  { background: linear-gradient(90deg, #f59e0b, #f97316); }
.metric-card.indigo::before { background: linear-gradient(90deg, #6366f1, #3b82f6); }
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 8px 40px rgba(59,130,246,0.18); }
.metric-value { font-size: 2.1rem; font-weight: 800; letter-spacing: -1px; }
.metric-label { font-size: 0.74rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.2px; margin-top: 0.3rem; }
.metric-delta { font-size: 0.78rem; margin-top: 0.5rem; font-weight: 500; }
.delta-pos { color: #22c55e; }
.delta-neutral { color: #94a3b8; }

/* ── Hero */
.hero {
  background: linear-gradient(135deg, #071829 0%, #0a1e38 50%, #061525 100%);
  border: 1px solid #1e3a5f;
  border-radius: 24px;
  padding: 3rem 2.5rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  margin-bottom: 2rem;
}
.hero::before {
  content: '';
  position: absolute;
  width: 700px; height: 700px;
  background: radial-gradient(circle, rgba(59,130,246,0.07) 0%, transparent 70%);
  top: -250px; left: 50%; transform: translateX(-50%);
  pointer-events: none;
}
.hero-title {
  font-size: clamp(1.9rem, 4.5vw, 3.4rem);
  font-weight: 800;
  background: linear-gradient(135deg, #60a5fa 0%, #06b6d4 50%, #818cf8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.15;
  margin-bottom: 0.8rem;
}
.hero-subtitle {
  font-size: 1rem;
  color: var(--text-muted);
  max-width: 680px;
  margin: 0 auto 1.5rem;
  line-height: 1.7;
}
.badge-row { display: flex; justify-content: center; gap: 0.75rem; flex-wrap: wrap; }
.badge {
  display: inline-flex; align-items: center; gap: 0.4rem;
  border-radius: 100px; padding: 0.45rem 1.1rem;
  font-size: 0.82rem; font-weight: 600;
}
.badge-green { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3); color: #22c55e; }
.badge-blue  { background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.3); color: #60a5fa; }
.badge-amber { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.3); color: #f59e0b; }

/* ── Section Headers */
.section-title {
  font-size: 1.4rem; font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 0.2rem;
}
.section-sub { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1.4rem; }

/* ── Pipeline Steps */
.pipeline-step {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.1rem 1.2rem;
  display: flex; align-items: flex-start; gap: 1rem;
  margin-bottom: 0.65rem;
  transition: border-color 0.2s, transform 0.2s;
}
.pipeline-step:hover { border-color: #3b82f6; transform: translateX(3px); }
.step-number {
  background: linear-gradient(135deg, #3b82f6, #06b6d4);
  border-radius: 8px; width: 34px; height: 34px;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.88rem; flex-shrink: 0;
}
.step-title { font-weight: 600; color: var(--text-primary); margin-bottom: 0.15rem; font-size: 0.95rem; }
.step-desc  { font-size: 0.80rem; color: var(--text-muted); line-height: 1.55; }

/* ── Info Boxes */
.info-box {
  background: rgba(59,130,246,0.07);
  border: 1px solid rgba(59,130,246,0.2);
  border-radius: 10px; padding: 1rem 1.2rem;
  font-size: 0.86rem; color: var(--text-muted); line-height: 1.65;
}
.success-box {
  background: rgba(34,197,94,0.07);
  border: 1px solid rgba(34,197,94,0.25);
  border-radius: 10px; padding: 1rem 1.2rem;
  font-size: 0.86rem; color: #86efac; line-height: 1.65;
}
.warning-box {
  background: rgba(245,158,11,0.07);
  border: 1px solid rgba(245,158,11,0.25);
  border-radius: 10px; padding: 1rem 1.2rem;
  font-size: 0.86rem; color: #fcd34d; line-height: 1.65;
}

/* ── Streamlit Overrides */
.stButton > button {
  background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
  color: white !important; border: none !important;
  border-radius: 10px !important; padding: 0.6rem 1.5rem !important;
  font-weight: 600 !important; font-family: 'Inter', sans-serif !important;
  transition: opacity 0.2s, transform 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-2px) !important; }
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, #3b82f6, #06b6d4) !important;
}
div[data-testid="stFileUploader"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
[data-testid="stMetric"] { background: var(--bg-card); border-radius: 10px; padding: 0.8rem; border: 1px solid var(--border); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(15,29,50,0.95)",
    plot_bgcolor="rgba(15,29,50,0.95)",
    font=dict(family="Inter", color="#94a3b8", size=12),
    xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f", linecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f", linecolor="#1e3a5f"),
    legend=dict(bgcolor="rgba(15,29,50,0.9)", bordercolor="#1e3a5f", borderwidth=1),
    margin=dict(t=45, b=40, l=55, r=25),
)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED: DEMO DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_demo_data():
    from sample_data import generate_demo_timeseries, get_demo_predictions
    df   = generate_demo_timeseries(n_days=365)
    pred = get_demo_predictions(df)
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# CACHED: REAL MODELS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load LGB + XGB + per-station models if present."""
    try:
        import lightgbm as lgb
        import xgboost as xgb

        lgb_path = "lgb_model.txt"
        xgb_path = "xgb_model.json"
        if not os.path.exists(lgb_path) or not os.path.exists(xgb_path):
            return None

        lgb_model = lgb.Booster(model_file=lgb_path)
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(xgb_path)

        WORST = [328, 242, 1, 142, 197, 290, 252, 219, 309, 213]
        station_models = {}
        for sid in WORST:
            p = f"lgb_station_{sid}.txt"
            if os.path.exists(p):
                station_models[sid] = lgb.Booster(model_file=p)

        # Try pickle metadata
        import pickle
        if os.path.exists("model_meta.pkl"):
            with open("model_meta.pkl", "rb") as f:
                meta = pickle.load(f)
            return lgb_model, xgb_model, station_models, \
                   meta["scales"], meta["weights"], meta["stats"], meta["enc"]

        return lgb_model, xgb_model, station_models, {}, \
               {"lgb": 1.0, "xgb": 0.0}, None, None

    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — model status & KGE metrics only (navigation via tabs)
# ─────────────────────────────────────────────────────────────────────────────
models = load_models()
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.2rem;'>
      <div style='font-size:2.5rem;'>🌊</div>
      <div style='font-size:1rem; font-weight:700; color:#f1f5f9; margin-top:0.4rem;'>Flood Prediction AI</div>
      <div style='font-size:0.70rem; color:#475569; margin-top:0.2rem;'>Ganga-Brahmaputra Basin · 2026</div>
    </div>
    """, unsafe_allow_html=True)

    if models is not None:
        n_stations = len(models[2])
        st.markdown(f"""
        <div class='success-box' style='text-align:center; margin-bottom:1rem;'>
          ✅ <b>Models Loaded</b><br>
          <span style='font-size:0.76rem; color:#86efac;'>LGB + XGB global<br>{n_stations} station specialists</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='warning-box' style='text-align:center; margin-bottom:1rem;'>
          ⚡ <b>Demo Mode</b><br>
          <span style='font-size:0.76rem;'>Add model files to enable live predictions</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.73rem; color:#475569;'>
      <div style='font-weight:600; color:#64748b; margin-bottom:0.4rem;
           text-transform:uppercase; letter-spacing:1px;'>KGE Results</div>
      <div style='margin-bottom:0.22rem;'>🎯 Val KGE <b style='color:#22c55e; font-size:0.88rem;'>0.999875</b></div>
      <div style='margin-bottom:0.22rem;'>📈 r &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b style='color:#60a5fa;'>0.9999</b></div>
      <div style='margin-bottom:0.22rem;'>📊 alpha &nbsp; <b style='color:#60a5fa;'>0.9999</b></div>
      <div style='margin-bottom:0.22rem;'>⚖️ beta &nbsp;&nbsp; <b style='color:#60a5fa;'>1.0000</b></div>
      <div style='margin-bottom:0.22rem;'>🔢 NSE &nbsp;&nbsp;&nbsp; <b style='color:#60a5fa;'>0.999827</b></div>
      <hr style='border-color:#1e3a5f; margin:0.6rem 0;'>
      <div style='font-weight:600; color:#64748b; margin-bottom:0.4rem;
           text-transform:uppercase; letter-spacing:1px;'>Architecture</div>
      <div style='margin-bottom:0.18rem;'>🌲 LightGBM (global)</div>
      <div style='margin-bottom:0.18rem;'>🌳 XGBoost (global)</div>
      <div style='margin-bottom:0.18rem;'>🔬 10 station specialists</div>
      <div style='margin-bottom:0.18rem;'>📡 367 gauge stations</div>
      <div style='margin-bottom:0.18rem;'>🔧 94 engineered features</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS — always-visible navigation (works at every screen size)
# ─────────────────────────────────────────────────────────────────────────────
tab_home, tab_predict, tab_explorer, tab_how = st.tabs(
    ["🏠 Home", "🔮 Predict", "📊 Explorer", "🧠 How It Works"]
)

with tab_home:
    st.markdown("""
    <div class='hero'>
      <div class='hero-title'>Flood Prediction AI<br>Ganga-Brahmaputra Basin</div>
      <div class='hero-subtitle'>
        Predicting next-day (t+1) river streamflow at 367 Indian gauge stations using
        an ensemble of LightGBM &amp; XGBoost — trained for the IIIT Lucknow
        StreamFlow Prediction Challenge 2026.
      </div>
      <div class='badge-row'>
        <div class='badge badge-green'>🏆 Val KGE 0.999875</div>
        <div class='badge badge-blue'>📡 367 Gauge Stations</div>
        <div class='badge badge-amber'>🌧 Ganga-Brahmaputra Basin</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class='metric-card green'>
          <div class='metric-value' style='color:#22c55e;'>0.999875</div>
          <div class='metric-label'>Validation KGE</div>
          <div class='metric-delta delta-pos'>+0.002615 vs baseline</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='metric-card blue'>
          <div class='metric-value' style='color:#60a5fa;'>367</div>
          <div class='metric-label'>Gauge Stations</div>
          <div class='metric-delta delta-neutral'>Single river basin</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='metric-card indigo'>
          <div class='metric-value' style='color:#818cf8;'>94</div>
          <div class='metric-label'>Features Used</div>
          <div class='metric-delta delta-neutral'>Flow + Rain + Soil</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class='metric-card amber'>
          <div class='metric-value' style='color:#f59e0b;'>3.2M</div>
          <div class='metric-label'>Training Rows</div>
          <div class='metric-delta delta-neutral'>Daily observations</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Demo chart
    with st.spinner("Generating demo…"):
        demo = load_demo_data()

    st.markdown("<div class='section-title'>📈 Live Demo — Seasonal Streamflow</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>365-day simulated flow for 8 Indian river stations with model predictions</div>", unsafe_allow_html=True)

    station_names = demo["station_name"].unique().tolist()
    sel = st.selectbox("Select Station", station_names, key="home_sta")
    sdf = demo[demo["station_name"] == sel].sort_values("day_of_year")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sdf["day_of_year"], y=sdf["true_tomorrow"],
        name="Observed Flow",
        line=dict(color="#3b82f6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=sdf["day_of_year"], y=sdf["prediction"],
        name="Prediction",
        line=dict(color="#22c55e", width=2.5, dash="dot"),
    ))
    fig.add_trace(go.Bar(
        x=sdf["day_of_year"],
        y=sdf["antecedent_rain_3d_sum"] * 6,
        name="Rainfall × 6 (proxy)",
        marker=dict(color="rgba(6,182,212,0.35)"),
        yaxis="y2",
    ))
    fig.add_vrect(x0=150, x1=270,
                  fillcolor="rgba(239,68,68,0.04)", line_width=0,
                  annotation_text="Peak Monsoon",
                  annotation_position="top left")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Streamflow Forecast — {sel}",
        xaxis_title="Day of Year",
        yaxis_title="Flow (m³/s)",
        yaxis2=dict(title="Rain proxy", overlaying="y", side="right",
                    showgrid=False, gridcolor="rgba(0,0,0,0)"),
        height=420,
        hovermode="x unified",
    )
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0))
    st.plotly_chart(fig, use_container_width=True)

    obs  = sdf["true_tomorrow"].values
    pred = sdf["prediction"].values
    r    = float(np.corrcoef(obs, pred)[0, 1])
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("🔺 Peak Flow", f"{obs.max():,.0f} m³/s")
    mc2.metric("〰 Mean Flow", f"{obs.mean():,.0f} m³/s")
    mc3.metric("📐 Pearson r", f"{r:.4f}")
    mc4.metric("🌧 Rain Days", f"{(sdf['antecedent_rain_3d_sum'] > 5).sum()}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Challenge context
    st.markdown("<div class='section-title'>🎯 The Challenge</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
      <b>Why is this hard?</b><br>
      The persistence baseline — simply predicting tomorrow's flow equals today's —
      already scores <b>KGE = 0.9978</b>. Rivers change slowly, so this naive approach is
      deceptively strong. The real challenge is building a model that captures the
      <b>rapid, nonlinear rise that precedes a flood event</b>, using upstream rainfall signals,
      soil saturation, and antecedent flow conditions — exactly when persistence fails most.<br><br>
      A model that predicts average flow well but collapses during flood peaks is useless for
      early warning. Submissions are evaluated using <b>Kling-Gupta Efficiency (KGE)</b>, a metric
      designed specifically to penalise errors in timing, variability, and volume simultaneously.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🚀 Key Innovations</div>", unsafe_allow_html=True)

    innovations = [
        ("1", "Delta-Target Formulation",
         "Predicts log₁ₚ(Q_tomorrow) − log₁ₚ(Q_today) instead of raw flow. "
         "Reduces target variance ~10×, forcing the model to focus on flood-onset dynamics rather than trivial autocorrelation."),
        ("2", "Flood-Weighted Loss",
         "weight = 1 + 4 × clip(Q/Q̄, 0, 5). High-flow rows receive up to 5× more gradient signal — "
         "flood peak prediction becomes a first-class objective."),
        ("3", "Per-Station Specialist Models",
         "10 dedicated LightGBM models for the hardest stations (KGE < 0.991). "
         "Final prediction = 50% global blend + 50% specialist."),
        ("4", "Scipy Ensemble Blending",
         "LightGBM + XGBoost predictions blended via SLSQP optimisation with 40 random restarts, "
         "maximising KGE directly on the validation set."),
        ("5", "Per-Station Bias Correction",
         "Multiplicative scale factor fitted per station on validation residuals. "
         "Clipped to [0.5, 2.0] for stability, [0.2, 5.0] for worst stations."),
    ]
    for num, title, desc in innovations:
        st.markdown(f"""
        <div class='pipeline-step'>
          <div class='step-number'>{num}</div>
          <div>
            <div class='step-title'>{title}</div>
            <div class='step-desc'>{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


with tab_predict:
    st.markdown("<div class='section-title'>🔮 Run Predictions</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Upload test CSV → get next-day streamflow predictions instantly</div>",
                unsafe_allow_html=True)


    if models is None:
        st.markdown("""
        <div class='warning-box'>
          ⚠️ <b>Model files not found.</b> Place <code>lgb_model.txt</code>, <code>xgb_model.json</code>,
          and per-station <code>lgb_station_*.txt</code> files in the app directory.<br>
          Demo prediction (physics-inspired baseline) will be used instead.
        </div>
        """, unsafe_allow_html=True)
    else:
        n_sta = len(models[2])
        st.markdown(f"""
        <div class='success-box'>
          ✅ <b>Full model loaded</b> — LightGBM global + XGBoost global + {n_sta} station specialists ready.
          Upload any test CSV to get real predictions.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📤 Upload & Predict", "📋 Column Reference"])

    with tab1:
        uploaded = st.file_uploader(
            "Upload CSV (test_flood.csv format)",
            type=["csv"],
        )

        if uploaded is not None:
            with st.spinner("Parsing CSV…"):
                try:
                    test_df = pd.read_csv(uploaded)
                    st.success(f"✅ Loaded **{len(test_df):,} rows × {test_df.shape[1]} cols**")
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")
                    st.stop()

            st.dataframe(test_df.head(8), use_container_width=True, height=220)

            req = ["row_id", "streamflow_today_cumecs", "day_of_year", "month",
                   "antecedent_rain_3d_sum", "soil_saturation_score"]
            missing = [c for c in req if c not in test_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
                st.stop()

            lbl = "🚀 Run Full Prediction" if models else "🚀 Run Demo Prediction"
            if st.button(lbl, use_container_width=True):
                prog  = st.progress(0.0)
                stat  = st.empty()

                if models is None:
                    # Physics-inspired fallback
                    for msg, pct in [("Preprocessing…", 0.2), ("Computing features…", 0.5),
                                     ("Running inference…", 0.8), ("Done!", 1.0)]:
                        stat.markdown(f"<div style='color:#94a3b8;font-size:0.85rem;'>{msg}</div>",
                                      unsafe_allow_html=True)
                        prog.progress(pct)
                        time.sleep(0.3)

                    q    = test_df["streamflow_today_cumecs"].values
                    rain = test_df.get("antecedent_rain_3d_sum",
                                       pd.Series(0, index=test_df.index)).values
                    sat  = test_df.get("soil_saturation_score",
                                       pd.Series(0, index=test_df.index)).values
                    rng  = np.random.default_rng(0)
                    sig  = np.log1p(rain) * (1 + sat * 0.5)
                    delta = rng.normal(0, 0.008, len(q)) + \
                            0.004 * (sig - sig.mean()) / (sig.std() + 1)
                    pred  = np.expm1(np.log1p(q) + delta).clip(0)
                    result = pd.DataFrame({
                        "row_id": test_df["row_id"].values,
                        "streamflow_tomorrow_cumecs": pred.round(4),
                    })
                else:
                    lgb_m, xgb_m, sta_m, scales, wts, stats, enc = models

                    steps = [
                        ("Assigning station IDs…",          0.08),
                        ("Merging station statistics…",     0.18),
                        ("Adding lag features…",            0.32),
                        ("Engineering 94 features…",        0.50),
                        ("LightGBM inference…",             0.65),
                        ("XGBoost inference…",              0.78),
                        ("Blending models…",                0.85),
                        ("Per-station specialists…",        0.92),
                        ("Bias correction…",                0.97),
                        ("Done!",                           1.00),
                    ]

                    def upd(msg, pct):
                        stat.markdown(
                            f"<div style='color:#94a3b8;font-size:0.85rem;'>⚙️ {msg}</div>",
                            unsafe_allow_html=True)
                        prog.progress(pct)

                    from utils import run_prediction_pipeline
                    result, _ = run_prediction_pipeline(
                        test_df.copy(), lgb_m, xgb_m, sta_m, scales,
                        wts, stats, enc, progress_cb=upd,
                    )

                stat.empty()
                prog.empty()

                st.success(f"✅ **{len(result):,} predictions ready!**")
                st.dataframe(result.head(25), use_container_width=True)

                ca, cb, cc = st.columns(3)
                ca.metric("Min", f"{result['streamflow_tomorrow_cumecs'].min():.2f} m³/s")
                cb.metric("Mean", f"{result['streamflow_tomorrow_cumecs'].mean():.2f} m³/s")
                cc.metric("Max", f"{result['streamflow_tomorrow_cumecs'].max():.2f} m³/s")

                fig_d = go.Figure(go.Histogram(
                    x=result["streamflow_tomorrow_cumecs"], nbinsx=60,
                    marker=dict(color="rgba(59,130,246,0.75)",
                                line=dict(color="#1d4ed8", width=0.5)),
                ))
                fig_d.update_layout(**PLOTLY_LAYOUT, height=260,
                                    xaxis_title="Streamflow (m³/s)", yaxis_title="Count",
                                    title="Prediction Distribution")
                st.plotly_chart(fig_d, use_container_width=True)

                st.download_button(
                    "⬇️ Download submission.csv",
                    data=result.to_csv(index=False).encode(),
                    file_name="submission.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    with tab2:
        st.markdown("### Required Columns")
        cols_info = {
            "row_id":                        "int32 — unique identifier",
            "streamflow_today_cumecs":       "float32 — today's observed flow (m³/s)",
            "streamflow_anomaly_zscore":     "float32 — z-score of flow anomaly",
            "flow_rate_of_change":           "float32 — Δ flow per day",
            "flow_velocity_km_per_day":      "float32 — flow velocity",
            "antecedent_rain_3d_sum":        "float32 — 3-day rainfall sum (mm)",
            "antecedent_rain_7d_sum":        "float32 — 7-day rainfall sum",
            "antecedent_rain_15d_sum":       "float32 — 15-day rainfall sum",
            "antecedent_rain_30d_sum":       "float32 — 30-day rainfall sum",
            "antecedent_rain_60d":           "float32 — 60-day rainfall",
            "antecedent_rain_ewm":           "float32 — EWM-smoothed rainfall",
            "rainfall_anomaly_zscore":       "float32 — rainfall z-score",
            "soil_saturation_score":         "float32 — [0,1] soil moisture proxy",
            "antecedent_saturation_interaction": "float32",
            "is_post_monsoon_saturated":     "int8 — binary flag",
            "monsoon_cumulative_rain":       "float32 — seasonal rain total",
            "monsoon_intensity":             "float32",
            "upstream_rain_mean_scaled":     "float32",
            "upstream_rain_weighted_scaled": "float32",
            "upstream_rain_lagged_dist_sink":"float32",
            "dist_to_outlet_scaled":         "float32 — basin static feature",
            "upstream_area_scaled":          "float32 — basin static feature",
            "slope_scaled":                  "float32 — basin static feature",
            "slope_uav_scaled":              "float32 — basin static feature",
            "forest_cover_scaled":           "float32 — land cover",
            "urban_cover_scaled":            "float32 — land cover",
            "rain_soilmoisture_interaction": "float32",
            "rain_urban_interaction":        "float32",
            "rain_slope_interaction":        "float32",
            "rain_basinsize_interaction":    "float32",
            "uparea_rain_interaction":       "float32",
            "day_of_year":                   "int16 — [1,365]",
            "month":                         "int8 — [1,12]",
        }
        st.dataframe(
            pd.DataFrame(list(cols_info.items()), columns=["Column", "Type & Description"]),
            use_container_width=True, hide_index=True,
        )
        tmpl = pd.DataFrame({c: [0] for c in cols_info})
        tmpl["row_id"] = [1]; tmpl["day_of_year"] = [180]; tmpl["month"] = [6]
        st.download_button("⬇️ Download CSV Template",
                           tmpl.to_csv(index=False).encode(),
                           "test_flood_template.csv", "text/csv")


with tab_explorer:
    st.markdown("<div class='section-title'>📊 Model Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Interactive analysis of predictions, features, and station performance</div>",
                unsafe_allow_html=True)

    with st.spinner("Loading demo data…"):
        demo = load_demo_data()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🌊 Flow Analysis", "🌧 Rain vs Flow", "🗺 Station Correlation", "🔑 Feature Importance"])

    # ─── Tab 1: Flow Analysis
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            sta = st.selectbox("Station", demo["station_name"].unique(), key="exp_s")
        with c2:
            win = st.slider("Smoothing window (days)", 1, 14, 5, key="exp_w")

        sdf = demo[demo["station_name"] == sta].sort_values("day_of_year").copy()
        sdf["err"] = sdf["prediction"] - sdf["true_tomorrow"]
        sdf["ae"]  = sdf["err"].abs()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.70, 0.30],
                            subplot_titles=["Observed vs Predicted", "Residual Error"],
                            vertical_spacing=0.09)
        fig.add_trace(go.Scatter(
            x=sdf["day_of_year"], y=sdf["true_tomorrow"], name="Observed",
            line=dict(color="#3b82f6", width=1.8),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.06)"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sdf["day_of_year"],
            y=sdf["prediction"].rolling(win, center=True, min_periods=1).mean(),
            name="Predicted", line=dict(color="#22c55e", width=2.5, dash="dot")), row=1, col=1)
        colors = ["#ef4444" if e > 0 else "#3b82f6" for e in sdf["err"]]
        fig.add_trace(go.Bar(
            x=sdf["day_of_year"], y=sdf["err"], name="Error",
            marker=dict(color=colors, opacity=0.6)), row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#475569", row=2, col=1)
        fig.update_layout(**PLOTLY_LAYOUT, height=580, hovermode="x unified",
                          yaxis_title="Flow (m³/s)", yaxis2_title="Error")
        st.plotly_chart(fig, use_container_width=True)

        # Scatter
        mx = max(sdf["true_tomorrow"].max(), sdf["prediction"].max()) * 1.05
        fig_s = go.Figure()
        fig_s.add_trace(go.Scattergl(
            x=sdf["true_tomorrow"], y=sdf["prediction"], mode="markers",
            marker=dict(color=sdf["ae"], colorscale="Turbo", size=5,
                        opacity=0.65, showscale=True,
                        colorbar=dict(title="Abs Err", thickness=12)),
        ))
        fig_s.add_trace(go.Scatter(x=[0, mx], y=[0, mx],
                                   line=dict(color="#ef4444", width=1, dash="dot"),
                                   name="Perfect fit"))
        fig_s.update_layout(**PLOTLY_LAYOUT, height=380,
                             xaxis_title="Observed (m³/s)", yaxis_title="Predicted (m³/s)",
                             title="Scatter: Predicted vs Observed")
        st.plotly_chart(fig_s, use_container_width=True)

    # ─── Tab 2: Rain vs Flow
    with tab2:
        sel_stas = st.multiselect("Stations",
                                   demo["station_name"].unique().tolist(),
                                   default=demo["station_name"].unique().tolist()[:5],
                                   key="rv")
        palette = ["#3b82f6","#22c55e","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#ec4899","#14b8a6"]

        fig_r = go.Figure()
        for i, s in enumerate(sel_stas or demo["station_name"].unique()[:5]):
            sd = demo[demo["station_name"] == s].sort_values("day_of_year")
            fig_r.add_trace(go.Scattergl(
                x=sd["antecedent_rain_3d_sum"], y=sd["streamflow_today_cumecs"],
                mode="markers", name=s,
                marker=dict(color=palette[i % len(palette)], size=4, opacity=0.5)))
        fig_r.update_layout(**PLOTLY_LAYOUT, height=400,
                             xaxis_title="3-day Rainfall (mm)", yaxis_title="Streamflow (m³/s)",
                             title="Rainfall–Streamflow Relationship by Station")
        st.plotly_chart(fig_r, use_container_width=True)

        # Monthly boxplot
        month_lbl = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig_box = go.Figure()
        for m in range(1, 13):
            vals = demo[demo["month"] == m]["streamflow_today_cumecs"].values
            if len(vals) == 0: continue
            clr = "#ef4444" if m in [7,8,9] else "#f59e0b" if m in [6,10] else "#3b82f6"
            fig_box.add_trace(go.Box(y=vals, name=month_lbl[m-1],
                                     marker_color=clr, boxmean=True))
        fig_box.update_layout(**PLOTLY_LAYOUT, height=360, showlegend=False,
                              yaxis_title="Flow (m³/s)", title="Monthly Streamflow Distribution")
        st.plotly_chart(fig_box, use_container_width=True)

    # ─── Tab 3: Station Correlation
    with tab3:
        records = []
        for s in demo["station_name"].unique():
            sd   = demo[demo["station_name"] == s]
            obs  = sd["true_tomorrow"].values
            pred = sd["prediction"].values
            r    = float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 2 else 0
            records.append({"Station": s, "Pearson r": round(r, 4),
                             "Mean Flow": round(obs.mean(), 1),
                             "Peak Flow": round(obs.max(), 1)})
        rdf = pd.DataFrame(records).sort_values("Pearson r")

        fig_kge = go.Figure(go.Bar(
            x=rdf["Pearson r"], y=rdf["Station"], orientation="h",
            marker=dict(
                color=["#22c55e" if r > 0.99 else "#f59e0b" if r > 0.95 else "#ef4444"
                       for r in rdf["Pearson r"]],
                opacity=0.85),
        ))
        fig_kge.add_vline(x=0.99, line_dash="dot", line_color="#94a3b8",
                          annotation_text="0.99 target", annotation_font_color="#94a3b8")
        fig_kge.update_layout(**PLOTLY_LAYOUT, height=380,
                              xaxis_title="Pearson r",
                              title="Per-Station Prediction Correlation")
        fig_kge.update_xaxes(range=[0.94, 1.005])
        st.plotly_chart(fig_kge, use_container_width=True)
        st.dataframe(rdf.sort_values("Pearson r", ascending=False).reset_index(drop=True),
                     use_container_width=True, hide_index=True)

    # ─── Tab 4: Feature Importance
    with tab4:
        st.markdown("##### Top Features by Gain — from LightGBM Training")
        features = [
            ("rel_rate",                    100.0, "Lag/Temporal"),
            ("flow_rate_of_change",          9.2,  "Lag/Temporal"),
            ("preflood_alarm",               3.8,  "Interaction"),
            ("flow_accel_sign",              1.5,  "Lag/Temporal"),
            ("flow_norm_station",            1.3,  "Flow"),
            ("streamflow_anomaly_zscore",    1.2,  "Flow"),
            ("log_flow",                     1.1,  "Flow"),
            ("streamflow_today_cumecs",      1.0,  "Flow"),
            ("log_flow_sq",                  0.9,  "Flow"),
            ("station_p90_flow",             0.85, "Station"),
            ("flow_lag_1",                   0.80, "Lag/Temporal"),
            ("flow_log_d1",                  0.75, "Lag/Temporal"),
            ("log_flow_lag_1",               0.70, "Lag/Temporal"),
            ("flow_spike_ratio",             0.65, "Lag/Temporal"),
            ("flow_roll_mean_3d",            0.60, "Lag/Temporal"),
            ("station_std_flow",             0.55, "Station"),
            ("doy_cos",                      0.50, "Temporal"),
            ("doy_sin",                      0.48, "Temporal"),
            ("logflow_x_sat",                0.45, "Interaction"),
            ("flow_lag1_ratio",              0.42, "Lag/Temporal"),
            ("slope_uav_scaled",             0.40, "Station"),
            ("is_receding",                  0.38, "Lag/Temporal"),
            ("flow_roll_max_7d",             0.35, "Lag/Temporal"),
            ("antecedent_rain_60d",          0.33, "Rain"),
            ("week_of_year",                 0.30, "Temporal"),
        ]
        cat_color = {"Lag/Temporal":"#ef4444","Rain":"#3b82f6",
                     "Station":"#8b5cf6","Interaction":"#22c55e",
                     "Flow":"#06b6d4","Temporal":"#f59e0b"}
        fdf = pd.DataFrame(features, columns=["Feature","Gain","Category"]).sort_values("Gain")

        fig_imp = go.Figure()
        for cat, clr in cat_color.items():
            sub = fdf[fdf["Category"] == cat]
            if sub.empty: continue
            fig_imp.add_trace(go.Bar(
                x=sub["Gain"], y=sub["Feature"], orientation="h",
                name=cat, marker=dict(color=clr, opacity=0.85)))
        fig_imp.update_layout(**PLOTLY_LAYOUT, height=640, barmode="stack",
                              xaxis_title="Relative Gain",
                              title="Top 25 Features by Gain (LightGBM)")
        st.plotly_chart(fig_imp, use_container_width=True)


with tab_how:
    st.markdown("<div class='section-title'>🧠 How It Works</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>The full pipeline behind the prediction model</div>",
                unsafe_allow_html=True)

    # Pipeline
    st.markdown("### 🔄 Full Pipeline")
    steps = [
        ("📂", "Load Data",
         "2.57M train + 642K test rows across 367 stations. Float32 dtype casting for memory efficiency."),
        ("🗂", "Station Identification",
         "Stations fingerprinted via 6 static basin features. LabelEncoder assigns integer IDs deterministically."),
        ("📊", "Station Statistics",
         "Per-station mean, std, and p90 of flow computed from training data and merged as contextual features."),
        ("⏳", "Lag Features",
         "7 flow lags + 3 rain lags + rolling mean/max/std over 3d and 7d windows — computed per station group."),
        ("🔧", "Feature Engineering",
         "94 total features: log transforms, flow derivatives (d1, d2), rain ratios, monsoon flags, pre-flood alarm, interactions."),
        ("🎯", "Delta Target",
         "Target = log₁ₚ(Q_tomorrow) − log₁ₚ(Q_today). Variance drops ~10×. Model forced to learn dynamics not autocorrelation."),
        ("⚖️", "Flood-Weighted Loss",
         "weight = 1 + 4×clip(Q/Q̄, 0, 5). Flood rows up to 5× gradient signal."),
        ("🌲", "LightGBM Global",
         "511 leaves, lr=0.03, 6000 trees, GPU. Best iter 691. Val KGE 0.999832."),
        ("🌳", "XGBoost Global",
         "depth=9, 511 leaves, CUDA. Best iter 1120. Val KGE 0.999808."),
        ("🔬", "Per-Station Specialists",
         "10 dedicated LGB models for stations with KGE < 0.991. Final = 50% global + 50% specialist."),
        ("⚗️", "Scipy Blend",
         "SLSQP optimisation, 40 restarts, maximising KGE. Found LGB=1.0, XGB≈0.0 optimal."),
        ("🔩", "Bias Correction",
         "Per-station scale ∈ [0.5, 2.0] fitted on validation residuals. Final val KGE → 0.999875."),
        ("📈", "Reconstruction",
         "Q_pred = expm₁(log₁ₚ(Q_today) + Δ_pred). Clipped to [0, ∞)."),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div class='pipeline-step'>
          <div class='step-number'>{icon}</div>
          <div>
            <div class='step-title'>{title}</div>
            <div class='step-desc'>{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Delta-Target — Visualised")

    with st.spinner("Generating…"):
        dd  = load_demo_data()
    ex  = dd[dd["station_name"] == dd["station_name"].iloc[0]].sort_values("day_of_year").head(120)
    obs = ex["true_tomorrow"].values
    prd = ex["prediction"].values
    doy = ex["day_of_year"].values
    log_obs  = np.log1p(obs)
    log_pred = np.log1p(prd)
    delta_obs  = np.diff(log_obs,  prepend=log_obs[0])
    delta_pred = np.diff(log_pred, prepend=log_pred[0])

    fig_d = make_subplots(rows=2, cols=2,
                          subplot_titles=["Raw Flow (Q_tomorrow)",
                                          "Log Space [log₁ₚ(Q)]",
                                          "Delta Target [Δlog Q] — what model predicts",
                                          "Reconstruction — back to m³/s"])
    fig_d.add_trace(go.Scatter(x=doy, y=obs, line=dict(color="#3b82f6",width=2), name="Observed"), row=1, col=1)
    fig_d.add_trace(go.Scatter(x=doy, y=log_obs, line=dict(color="#06b6d4",width=2), name="log Q"), row=1, col=2)
    fig_d.add_trace(go.Bar(x=doy, y=delta_obs,
                           marker=dict(color=["#ef4444" if d > 0 else "#3b82f6" for d in delta_obs], opacity=0.7),
                           name="Δlog Q"), row=2, col=1)
    fig_d.add_trace(go.Scatter(x=doy, y=obs, line=dict(color="#3b82f6",width=2), name="Observed"), row=2, col=2)
    fig_d.add_trace(go.Scatter(x=doy, y=prd, line=dict(color="#22c55e",width=2,dash="dot"), name="Reconstructed"), row=2, col=2)
    fig_d.update_layout(**PLOTLY_LAYOUT, height=540, showlegend=False,
                        title="The Delta-Target Formulation")
    st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("""
    <div class='info-box'>
      <b>Why this works:</b> Predicting log₁ₚ(Q_tomorrow) directly means 90% of training rows have
      "tomorrow ≈ today" (autocorrelation). The model fits these trivial cases and under-serves flood
      onset/recession days. By predicting <b>Δ = log₁ₚ(Q_tomorrow) − log₁ₚ(Q_today)</b>, stable days
      have Δ ≈ 0 (easy, ignored) and flood events have large Δ (hard, prioritised). Combined with
      flood-weighted loss, the model learns to predict the nonlinear dynamics that matter for KGE.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📐 KGE Component Analysis")

    components = ["r (timing)", "α (variability)", "β (bias)"]
    fig_sp = go.Figure()
    fill_rgba = ["rgba(34,197,94,0.08)", "rgba(59,130,246,0.08)", "rgba(148,163,184,0.08)"]
    for (vals, name, color), fill in zip([
        ([0.9999, 0.9999, 1.0000], "Our Model (0.999875)", "#22c55e"),
        ([0.9992, 0.9997, 0.9998], "Baseline LGB v8 (0.99726)", "#3b82f6"),
        ([0.9979, 1.0000, 0.9999], "Persistence (0.997882)", "#94a3b8"),
    ], fill_rgba):
        fig_sp.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=components + [components[0]],
            name=name, line=dict(color=color, width=2.5),
            fill="toself", fillcolor=fill,
        ))
    fig_sp.update_layout(
        **PLOTLY_LAYOUT, height=420,
        polar=dict(
            bgcolor="rgba(15,29,50,0.95)",
            radialaxis=dict(visible=True, range=[0.996, 1.001],
                            gridcolor="#1e3a5f", color="#94a3b8"),
            angularaxis=dict(gridcolor="#1e3a5f", color="#94a3b8"),
        ),
        title="KGE Component Comparison",
    )
    st.plotly_chart(fig_sp, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📦 Model Architecture")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("""
**LightGBM (Global)**
| Parameter | Value |
|-----------|-------|
| num_leaves | 511 |
| learning_rate | 0.03 |
| n_estimators | 6,000 |
| feature_fraction | 0.75 |
| bagging_fraction | 0.80 |
| lambda_l1 / l2 | 0.05 / 0.50 |
| device | GPU |
| Best iteration | 691 |
| **Val KGE** | **0.999832** |
""")
    with cb:
        st.markdown("""
**XGBoost (Global)**
| Parameter | Value |
|-----------|-------|
| max_depth | 9 |
| max_leaves | 511 |
| learning_rate | 0.03 |
| n_estimators | 6,000 |
| colsample_bytree | 0.75 |
| subsample | 0.80 |
| device | CUDA |
| Best iteration | 1,120 |
| **Val KGE** | **0.999808** |
""")
    st.markdown("""
> **Final blend:** Scipy found LGB=1.0, XGB≈0 optimal on validation.  
> Per-station specialists add +0.000007 overall but significantly improve the 10 hardest stations.  
> Final val KGE after bias correction: **0.999875**
""")
