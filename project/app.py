"""
Cardiovascular Disease Prediction – Streamlit App
===================================================
Supports BOTH Logistic Regression and Random Forest.
Select your model from the sidebar dropdown and predict!
Run:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioRisk AI | Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    *, html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #1a1a4e, #0f0c29);
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.10);
    }

    /* ── Model selector badge ── */
    .model-badge-lr {
        display:inline-block;
        background: linear-gradient(135deg,#f59e0b,#ef4444);
        border-radius: 20px;
        padding: 6px 18px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
        margin-bottom: 8px;
    }
    .model-badge-rf {
        display:inline-block;
        background: linear-gradient(135deg,#7c3aed,#3b82f6);
        border-radius: 20px;
        padding: 6px 18px;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
        margin-bottom: 8px;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-4px); }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Hero ── */
    .hero-header { text-align: center; padding: 2rem 1rem 1rem; }
    .hero-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    .hero-header p { font-size: 1rem; color: #94a3b8; }

    /* ── Prediction boxes ── */
    .prediction-high {
        background: linear-gradient(135deg,rgba(239,68,68,.2),rgba(239,68,68,.05));
        border: 2px solid rgba(239,68,68,.5);
        border-radius: 20px; padding: 2rem; text-align: center;
        animation: pulse-red 2s infinite;
    }
    .prediction-low {
        background: linear-gradient(135deg,rgba(52,211,153,.2),rgba(52,211,153,.05));
        border: 2px solid rgba(52,211,153,.5);
        border-radius: 20px; padding: 2rem; text-align: center;
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-red  { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.3)} 50%{box-shadow:0 0 20px 4px rgba(239,68,68,.15)} }
    @keyframes pulse-green{ 0%,100%{box-shadow:0 0 0 0 rgba(52,211,153,.3)} 50%{box-shadow:0 0 20px 4px rgba(52,211,153,.15)} }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.25rem; font-weight: 700; color: #a78bfa;
        margin: 1rem 0 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid rgba(167,139,250,.3);
    }

    /* ── ALL widget labels → white ── */
    label, .stSlider label, .stSelectbox label,
    .stToggle label, .stCheckbox label,
    .stTextInput label, .stNumberInput label,
    .stTextArea label, .stDateInput label,
    .stMultiSelect label, .stRadio label,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    .stSlider [data-testid="stWidgetLabel"],
    .stSlider [data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
    }

    /* Slider min/max value numbers */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"],
    .stSlider span { color: #ffffff !important; }

    /* Selectbox placeholder & option text - KEEP GREY */
    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] div { color: #cbd5e1 !important; }

    /* Toggle label text */
    .stToggle p, .stToggle span,
    [data-testid="stToggle"] p { color: #ffffff !important; }

    /* st.metric label, value, delta */
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] { color: #ffffff !important; }
    [data-testid="stMetricValue"]  { color: #ffffff !important; }

    /* Caption / small helper text below widgets */
    .stCaption, .stCaption p, small { color: #ffffff !important; }

    /* Bold markdown text inside the form */
    p strong, p b { color: #ffffff !important; }

    /* ── Buttons ── */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #7c3aed, #3b82f6);
        color: white; border: none; border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-size: 1.05rem; font-weight: 600;
        transition: opacity .2s, transform .1s;
    }
    .stButton > button:hover { opacity: .9; transform: translateY(-2px); }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,.04);
        border-radius: 12px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"]        { border-radius: 8px; color: #94a3b8; }
    .stTabs [aria-selected="true"]      { background: linear-gradient(135deg,#7c3aed,#3b82f6)!important; color:white!important; }

    /* ── Comparison table ── */
    .cmp-table { width:100%; border-collapse:collapse; }
    .cmp-table th { background:rgba(167,139,250,.15); color:#a78bfa; font-size:.8rem;
                    text-transform:uppercase; padding:10px 14px; text-align:left; }
    .cmp-table td { padding:10px 14px; border-bottom:1px solid rgba(255,255,255,.06); font-size:.9rem; }
    .cmp-table tr:hover td { background:rgba(255,255,255,.04); }
    .win  { color:#34d399; font-weight:700; }
    .lose { color:#94a3b8; }

    div.block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS & LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PLOT_DIR  = os.path.join(os.path.dirname(__file__), "plots")

MODEL_OPTIONS = {
    "🌳 Random Forest":        "random_forest",
    "📈 Logistic Regression":  "logistic_regression",
}

MODEL_COLORS = {
    "🌳 Random Forest":       "#7c3aed",
    "📈 Logistic Regression": "#f59e0b",
}

@st.cache_resource(show_spinner="Loading models …")
def load_all():
    if not os.path.exists(os.path.join(MODEL_DIR, "all_metrics.pkl")):
        return None
    return {
        "scaler":   joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
        "features": joblib.load(os.path.join(MODEL_DIR, "features.pkl")),
        "all_metrics": joblib.load(os.path.join(MODEL_DIR, "all_metrics.pkl")),
        "models": {
            "🌳 Random Forest":
                joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl")),
            "📈 Logistic Regression":
                joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl")),
        },
    }

artifacts = load_all()

# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-header">
        <h1>❤️ CardioRisk AI</h1>
        <p>Cardiovascular Disease Risk Predictor · Logistic Regression &amp; Random Forest</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# GUARD – models not trained yet
# ─────────────────────────────────────────────────────────────
if artifacts is None:
    st.error(
        "⚠️ **Models not found.**  \n"
        "Please run `python train_model.py` first, then refresh this page."
    )
    st.stop()

scaler      = artifacts["scaler"]
FEATURES    = artifacts["features"]
all_metrics = artifacts["all_metrics"]

# ─────────────────────────────────────────────────────────────
# SIDEBAR – project info only (model selector lives in the form)
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ CardioRisk AI")
    st.markdown("---")
    st.markdown(
        """
        <div style="color:#94a3b8;font-size:0.88rem;line-height:1.7;">
        Predict cardiovascular disease risk using two ML models.<br><br>
        🌳 <strong style="color:#a78bfa;">Random Forest</strong><br>
        📈 <strong style="color:#f59e0b;">Logistic Regression</strong><br><br>
        Select your model directly inside the <strong>Predict Risk</strong> tab.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        """
        <div style="color:#64748b;font-size:0.78rem;">
        📊 70,000 patient records<br>
        🔬 cardio_train.csv dataset<br>
        ⚠️ For educational use only
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_predict, tab_metrics, tab_compare, tab_importance, tab_about = st.tabs(
    ["🔍 Predict Risk", "📊 Model Performance", "⚖️ Model Comparison", "🌳 Feature Insights", "ℹ️ About"]
)

# ═════════════════════════════════════════════════════════════
#  TAB 1 – PREDICT RISK
# ═════════════════════════════════════════════════════════════
with tab_predict:

    # ── MODEL SELECTOR DROPDOWN (inside the form) ────────────
    st.markdown("<div class='section-header'>🤖 Select Model</div>", unsafe_allow_html=True)

    sel_col, info_col = st.columns([2, 3])
    with sel_col:
        selected_label = st.selectbox(
            "Choose ML Model for Prediction",
            list(MODEL_OPTIONS.keys()),
            index=0,
            key="form_model_select",
            help="Switch between Logistic Regression and Random Forest to compare results.",
        )

    # Derive active model variables from the in-form selection
    slug      = MODEL_OPTIONS[selected_label]
    model     = artifacts["models"][selected_label]
    m_color   = MODEL_COLORS[selected_label]
    m_metrics = all_metrics[
        "Random Forest" if "Forest" in selected_label else "Logistic Regression"
    ]

    with info_col:
        badge_class = "model-badge-rf" if "Forest" in selected_label else "model-badge-lr"
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:16px;
                background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);
                border-left:5px solid {m_color};
                border-radius:14px;padding:14px 20px;margin-top:22px;">
                <div style="font-size:2rem;">{'🌳' if 'Forest' in selected_label else '📈'}</div>
                <div>
                    <div style="font-size:1rem;font-weight:700;color:{m_color};">{selected_label}</div>
                    <div style="font-size:.8rem;color:#94a3b8;margin-top:2px;">
                        Accuracy: <strong>{m_metrics['accuracy']*100:.1f}%</strong> &nbsp;·&nbsp;
                        ROC-AUC: <strong>{m_metrics['roc_auc']:.4f}</strong> &nbsp;·&nbsp;
                        CV: <strong>{m_metrics['cv_mean']*100:.1f}%</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Enter Patient Details</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🧍 Demographics**")
        age    = st.slider("Age (years)", 18, 100, 50, 1, key="age")
        gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
        height = st.slider("Height (cm)", 100, 220, 165, 1, key="height")
        weight = st.slider("Weight (kg)", 30.0, 200.0, 70.0, 0.5, key="weight")

    with col2:
        st.markdown("**🩺 Blood Pressure**")
        ap_hi = st.slider("Systolic BP (ap_hi) mmHg",  60, 250, 120, 1, key="ap_hi")
        ap_lo = st.slider("Diastolic BP (ap_lo) mmHg", 40, 200,  80, 1, key="ap_lo")
        st.markdown("**🧪 Blood Tests**")
        cholesterol = st.selectbox(
            "Cholesterol Level", [1, 2, 3],
            format_func=lambda x: {1:"Normal", 2:"Above Normal", 3:"Well Above Normal"}[x],
            key="cholesterol",
        )
        gluc = st.selectbox(
            "Glucose Level", [1, 2, 3],
            format_func=lambda x: {1:"Normal", 2:"Above Normal", 3:"Well Above Normal"}[x],
            key="gluc",
        )

    with col3:
        st.markdown("**🚬 Lifestyle**")
        smoke  = st.toggle("Smoker",             value=False, key="smoke")
        alco   = st.toggle("Drinks Alcohol",      value=False, key="alco")
        active = st.toggle("Physically Active",   value=True,  key="active")

        bmi = weight / ((height / 100) ** 2)
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Calculated BMI", f"{bmi:.1f}")
        bmi_cat = (
            "Underweight 🪶" if bmi < 18.5 else
            "Normal ✅"      if bmi < 25   else
            "Overweight ⚠️"  if bmi < 30   else
            "Obese 🔴"
        )
        st.caption(f"Category: **{bmi_cat}**")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button(
        f"🔍 Predict with {selected_label}", use_container_width=True
    )

    if predict_btn:
        input_data = {
            "age": age, "gender": 1 if gender == "Female" else 2,
            "height": height, "weight": weight,
            "ap_hi": ap_hi, "ap_lo": ap_lo,
            "cholesterol": cholesterol, "gluc": gluc,
            "smoke": int(smoke), "alco": int(alco), "active": int(active),
            "bmi": round(bmi, 2),
        }
        input_df = pd.DataFrame([input_data])[FEATURES]
        input_sc = scaler.transform(input_df)

        prob       = model.predict_proba(input_sc)[0][1]
        prediction = model.predict(input_sc)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        res_col, gauge_col = st.columns(2)

        with res_col:
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="prediction-high">
                        <div style="font-size:1.4rem;font-weight:700;margin-bottom:.4rem;">🔴 High Cardiovascular Risk</div>
                        <div style="font-size:3rem;font-weight:800;color:#ef4444;margin:.5rem 0;">{prob*100:.1f}%</div>
                        <div style="color:#cbd5e1;font-size:.9rem;">
                            Predicted by <strong>{selected_label}</strong><br>
                            Risk indicators are elevated. Recommend clinical evaluation.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="prediction-low">
                        <div style="font-size:1.4rem;font-weight:700;margin-bottom:.4rem;">🟢 Low Cardiovascular Risk</div>
                        <div style="font-size:3rem;font-weight:800;color:#34d399;margin:.5rem 0;">{prob*100:.1f}%</div>
                        <div style="color:#cbd5e1;font-size:.9rem;">
                            Predicted by <strong>{selected_label}</strong><br>
                            Indicators appear within normal range. Keep up healthy habits!
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with gauge_col:
            gauge_color = "#ef4444" if prob > 0.5 else "#34d399"
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={"suffix":"%","font":{"size":36,"color":gauge_color}},
                delta={"reference":50,"valueformat":".1f"},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#94a3b8"},
                    "bar":{"color":gauge_color},
                    "bgcolor":"rgba(255,255,255,.05)",
                    "bordercolor":"rgba(255,255,255,.1)",
                    "steps":[
                        {"range":[0,30],  "color":"rgba(52,211,153,.15)"},
                        {"range":[30,60], "color":"rgba(251,191,36,.15)"},
                        {"range":[60,100],"color":"rgba(239,68,68,.15)"},
                    ],
                    "threshold":{"line":{"color":"white","width":3},"thickness":.9,"value":50},
                },
                title={"text":"Risk Probability","font":{"color":"#94a3b8","size":14}},
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                height=280, margin=dict(t=30,b=10,l=20,r=20),
            )
            st.plotly_chart(fig_g, use_container_width=True)

        # ── Also run the OTHER model for comparison ──
        other_label = [k for k in MODEL_OPTIONS if k != selected_label][0]
        other_model = artifacts["models"][other_label]
        other_prob  = other_model.predict_proba(input_sc)[0][1]
        other_pred  = other_model.predict(input_sc)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🔄 Both Models Side-by-Side</div>", unsafe_allow_html=True)

        cmp_c1, cmp_c2 = st.columns(2)
        for col, lbl, p, pred in [
            (cmp_c1, selected_label, prob,       prediction),
            (cmp_c2, other_label,    other_prob,  other_pred),
        ]:
            clr = "#ef4444" if pred == 1 else "#34d399"
            outcome = "High Risk 🔴" if pred == 1 else "Low Risk 🟢"
            with col:
                st.markdown(
                    f"""
                    <div style="border:1px solid rgba(255,255,255,.12);
                        border-left:4px solid {MODEL_COLORS[lbl]};
                        background:rgba(255,255,255,.04);
                        border-radius:14px;padding:18px 20px;text-align:center;">
                        <div style="font-size:.8rem;color:#94a3b8;text-transform:uppercase; margin-bottom:6px;">{lbl}</div>
                        <div style="font-size:2.2rem;font-weight:800;color:{clr};">{p*100:.1f}%</div>
                        <div style="font-size:.95rem;font-weight:600;color:{clr};">{outcome}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Risk factor cards ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🔎 Key Risk Factors</div>", unsafe_allow_html=True)

        risk_factors = []
        if ap_hi >= 140 or ap_lo >= 90:
            risk_factors.append(("🩸 High Blood Pressure", f"Systolic: {ap_hi}  Diastolic: {ap_lo}", "red"))
        if bmi >= 30:
            risk_factors.append(("⚖️ Obesity",     f"BMI: {bmi:.1f}", "red"))
        elif bmi >= 25:
            risk_factors.append(("⚖️ Overweight",  f"BMI: {bmi:.1f}", "orange"))
        if cholesterol >= 2:
            risk_factors.append(("🧪 High Cholesterol", "Above Normal" if cholesterol==2 else "Well Above Normal", "orange"))
        if gluc >= 2:
            risk_factors.append(("🍬 High Glucose",     "Above Normal" if gluc==2 else "Well Above Normal", "orange"))
        if smoke:   risk_factors.append(("🚬 Smoker",    "Active smoker",        "red"))
        if alco:    risk_factors.append(("🍺 Alcohol",  "Drinks alcohol",        "orange"))
        if not active: risk_factors.append(("🛋️ Sedentary", "Not physically active","orange"))
        if age >= 60: risk_factors.append(("📅 Age",    f"Age {age} – higher baseline risk", "orange"))

        if risk_factors:
            cols = st.columns(min(len(risk_factors), 3))
            for i, (title, detail, color) in enumerate(risk_factors):
                bc = "#ef4444" if color=="red" else "#f59e0b"
                bg = "rgba(239,68,68,.1)" if color=="red" else "rgba(245,158,11,.1)"
                with cols[i % 3]:
                    st.markdown(
                        f"""
                        <div style="border:1px solid {bc};background:{bg};
                            border-radius:12px;padding:14px 16px;margin-bottom:10px;">
                            <div style="font-weight:600;font-size:.95rem;">{title}</div>
                            <div style="color:#94a3b8;font-size:.82rem;">{detail}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.success("✅ No major individual risk factors detected.")

# ═════════════════════════════════════════════════════════════
#  TAB 2 – MODEL PERFORMANCE  (own selector so it's independent)
# ═════════════════════════════════════════════════════════════
with tab_metrics:
    perf_label = st.selectbox(
        "📊 View performance for model:",
        list(MODEL_OPTIONS.keys()),
        index=0,
        key="perf_model_select",
    )
    perf_slug     = MODEL_OPTIONS[perf_label]
    perf_metrics  = all_metrics[
        "Random Forest" if "Forest" in perf_label else "Logistic Regression"
    ]
    perf_color    = MODEL_COLORS[perf_label]

    st.markdown(
        f"<div class='section-header'>📊 {perf_label} – Performance Metrics</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in [
        (c1, "Accuracy",         f"{perf_metrics['accuracy']*100:.1f}%"),
        (c2, "ROC-AUC",          f"{perf_metrics['roc_auc']:.4f}"),
        (c3, "CV Score (5-fold)", f"{perf_metrics['cv_mean']*100:.1f}%"),
        (c4, "Test Samples",     f"{perf_metrics['n_test']:,}"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{val}</div>'
                f'<div class="metric-label">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    img1_col, img2_col = st.columns(2)

    conf_path = os.path.join(PLOT_DIR, f"confusion_matrix_{perf_slug}.png")
    roc_path  = os.path.join(PLOT_DIR, f"roc_curve_{perf_slug}.png")

    if os.path.exists(conf_path):
        with img1_col:
            st.image(conf_path, caption="Confusion Matrix", use_column_width=True)
    if os.path.exists(roc_path):
        with img2_col:
            st.image(roc_path, caption="ROC Curve", use_column_width=True)

# ═════════════════════════════════════════════════════════════
#  TAB 3 – MODEL COMPARISON
# ═════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("<div class='section-header'>⚖️ Logistic Regression vs Random Forest</div>", unsafe_allow_html=True)

    lr_m  = all_metrics["Logistic Regression"]
    rf_m  = all_metrics["Random Forest"]

    # ── Comparison table ──
    rows = [
        ("Accuracy",          f"{lr_m['accuracy']*100:.2f}%", f"{rf_m['accuracy']*100:.2f}%",
         lr_m['accuracy'] < rf_m['accuracy']),
        ("ROC-AUC",           f"{lr_m['roc_auc']:.4f}",       f"{rf_m['roc_auc']:.4f}",
         lr_m['roc_auc'] < rf_m['roc_auc']),
        ("CV Mean Accuracy",  f"{lr_m['cv_mean']*100:.2f}%",  f"{rf_m['cv_mean']*100:.2f}%",
         lr_m['cv_mean'] < rf_m['cv_mean']),
        ("CV Std Dev",        f"{lr_m['cv_std']*100:.2f}%",   f"{rf_m['cv_std']*100:.2f}%",
         lr_m['cv_std'] > rf_m['cv_std']),  # lower std is better for RF
    ]

    table_html = f"""<div style="overflow-x:auto;">
    <table class="cmp-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th>📈 Logistic Regression</th>
                <th>🌳 Random Forest</th>
            </tr>
        </thead>
        <tbody>"""
    for metric, lr_val, rf_val, rf_wins in rows:
        lr_cls = "win" if not rf_wins else "lose"
        rf_cls = "win" if rf_wins else "lose"
        lr_icon = " ✓" if not rf_wins else ""
        rf_icon = " ✓" if rf_wins else ""
        table_html += f"""
            <tr>
                <td><strong>{metric}</strong></td>
                <td class="{lr_cls}">{lr_val}{lr_icon}</td>
                <td class="{rf_cls}">{rf_val}{rf_icon}</td>
            </tr>"""
    table_html += "</tbody></table></div>"
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Side-by-side bar chart ──
    metrics_names = ["Accuracy", "ROC-AUC", "CV Mean"]
    lr_vals = [lr_m["accuracy"], lr_m["roc_auc"], lr_m["cv_mean"]]
    rf_vals = [rf_m["accuracy"], rf_m["roc_auc"], rf_m["cv_mean"]]

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name="📈 Logistic Regression", x=metrics_names, y=lr_vals,
        marker_color="#f59e0b", text=[f"{v:.3f}" for v in lr_vals],
        textposition="outside",
    ))
    fig_cmp.add_trace(go.Bar(
        name="🌳 Random Forest", x=metrics_names, y=rf_vals,
        marker_color="#7c3aed", text=[f"{v:.3f}" for v in rf_vals],
        textposition="outside",
    ))
    fig_cmp.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.02)",
        font_color="#e2e8f0",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(range=[0.6, 1.0], title="Score"),
        title="Model Metrics Comparison",
        height=400,
        margin=dict(t=50,b=20,l=20,r=20),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Combined ROC curves ──
    roc_cmp_path = os.path.join(PLOT_DIR, "roc_comparison.png")
    if os.path.exists(roc_cmp_path):
        st.image(roc_cmp_path, caption="ROC Curve – Both Models", use_column_width=True)

    # ── Feature cards ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🧠 Which Model Should You Use?</div>", unsafe_allow_html=True)
    fa, fb = st.columns(2)
    with fa:
        st.markdown(
            """
            <div style="border:1px solid #f59e0b;background:rgba(245,158,11,.08);
                border-radius:14px;padding:18px 20px;">
                <div style="font-size:1rem;font-weight:700;color:#f59e0b;margin-bottom:8px;">📈 Logistic Regression</div>
                <ul style="color:#cbd5e1;font-size:.88rem;padding-left:18px;margin:0;">
                    <li>Fast training &amp; inference</li>
                    <li>Highly interpretable (coefficients)</li>
                    <li>Good baseline model</li>
                    <li>Assumes linear decision boundary</li>
                    <li>Less sensitive to hyperparameters</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fb:
        st.markdown(
            """
            <div style="border:1px solid #7c3aed;background:rgba(124,58,237,.08);
                border-radius:14px;padding:18px 20px;">
                <div style="font-size:1rem;font-weight:700;color:#a78bfa;margin-bottom:8px;">🌳 Random Forest</div>
                <ul style="color:#cbd5e1;font-size:.88rem;padding-left:18px;margin:0;">
                    <li>Handles non-linear relationships</li>
                    <li>Captures feature interactions</li>
                    <li>Built-in feature importance</li>
                    <li>Robust to outliers</li>
                    <li>Generally higher accuracy</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════════════════════
#  TAB 4 – FEATURE INSIGHTS  (own selector)
# ═════════════════════════════════════════════════════════════
with tab_importance:
    fi_label = st.selectbox(
        "🌳 View feature importance for model:",
        list(MODEL_OPTIONS.keys()),
        index=0,
        key="fi_model_select",
    )
    fi_metrics = all_metrics[
        "Random Forest" if "Forest" in fi_label else "Logistic Regression"
    ]
    fi_color = MODEL_COLORS[fi_label]

    st.markdown(
        f"<div class='section-header'>🌳 Feature Insights – {fi_label}</div>",
        unsafe_allow_html=True,
    )

    fi = fi_metrics["feature_importances"]
    fi_df = (
        pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
        .sort_values("Importance", ascending=False)
    )

    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Viridis",
        labels={"Importance":"Importance Score","Feature":""},
        title=f"Feature Importance – {fi_label}",
    )
    fig_fi.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.02)",
        font_color="#e2e8f0",
        coloraxis_showscale=False,
        yaxis={"categoryorder":"total ascending"},
        margin=dict(l=10,r=10,t=50,b=10),
        height=420,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Top-5 radar
    top5 = fi_df.head(5)
    fig_radar = go.Figure(go.Scatterpolar(
        r=top5["Importance"].tolist() + [top5["Importance"].iloc[0]],
        theta=top5["Feature"].tolist() + [top5["Feature"].iloc[0]],
        fill="toself",
        fillcolor=f"rgba({','.join(str(int(x*255)) for x in [0.48,0.23,0.93])},0.25)",
        line=dict(color=fi_color, width=2),
        name="Top 5 Features",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,.03)",
            radialaxis=dict(visible=True, color="#94a3b8"),
            angularaxis=dict(color="#94a3b8"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        title="Top-5 Feature Radar",
        height=380,
        margin=dict(l=20,r=20,t=60,b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ═════════════════════════════════════════════════════════════
#  TAB 5 – ABOUT
# ═════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("<div class='section-header'>ℹ️ About This Project</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([2,1])
    with col_a:
        st.markdown(
            """
            ### CardioRisk AI – Cardiovascular Disease Prediction

            This project trains and compares **two ML classifiers** to predict the likelihood
            of cardiovascular disease (CVD) based on patient health metrics.

            #### 📂 Dataset
            - **Source**: `cardio_train.csv` (70 000 patient records)
            - **Target**: `cardio` – presence (1) or absence (0) of CVD
            - **Features**: age, gender, height, weight, blood pressure, cholesterol,
              glucose, smoking status, alcohol use, physical activity, BMI (derived)

            #### 🤖 Models
            | | Logistic Regression | Random Forest |
            |--|--|--|
            | Type | Linear | Ensemble (trees) |
            | Non-linearity | ❌ | ✅ |
            | Feature importance | Coefficient | Built-in |
            | Speed | ⚡ Very fast | 🐢 Slower |
            | Interpretability | ✅ High | ⚠️ Medium |

            #### ⚠️ Disclaimer
            > This tool is **for educational purposes only** and should **not** be used
            > as a substitute for professional medical advice, diagnosis, or treatment.
            """
        )
    with col_b:
        for icon, label, value in [
            ("🧠", "Models", "LR + Random Forest"),
            ("📦", "Stack", "scikit-learn · Streamlit · Plotly"),
            ("🗄️", "Dataset", "70 000 records"),
            ("🎯", "Target", "Cardiovascular Disease"),
        ]:
            st.markdown(
                f"""
                <div class="metric-card" style="margin-top:.8rem;">
                    <div style="font-size:1.8rem;margin-bottom:.4rem;">{icon}</div>
                    <div class="metric-label">{label}</div>
                    <div style="font-size:.88rem;font-weight:600;color:#a78bfa;margin-top:4px;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center;color:#475569;font-size:.78rem;padding:1rem;">
        ❤️ CardioRisk AI · Logistic Regression &amp; Random Forest ·
        Built with Streamlit &amp; scikit-learn · <em>Educational use only</em>
    </div>
    """,
    unsafe_allow_html=True,
)
