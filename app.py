import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Tech Track Recommender",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        color: white;
        box-shadow: 0 4px 14px rgba(102, 126, 234, 0.35);
        border: none;
    }
    div[data-testid="metric-container"] label {
        color: rgba(255,255,255,0.9) !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: white !important;
        font-weight: 800 !important;
        font-size: 1.9rem !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stRadio > label { display: none; }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        margin: 0.15rem 0;
        transition: background 0.2s;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.12);
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.02em;
        transition: all 0.25s;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.5);
        color: white;
    }

    h1 { color: #1e3c72; font-weight: 800; letter-spacing: -0.02em; }
    h2 { color: #2a5298; font-weight: 700; }
    h3 { color: #2a5298; font-weight: 600; }

    div[data-testid="stForm"] {
        background: #f8f9ff;
        border: 1px solid #e5e9f5;
        border-radius: 14px;
        padding: 1.5rem;
    }

    .q-section {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 1rem 0 0.6rem 0;
    }

    .chip {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        border: 1px solid #c7d2fe;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.82rem;
        margin: 0.15rem 0.2rem 0.15rem 0;
        font-weight: 500;
    }
    .chip-yes { background:#dcfce7; color:#166534; border-color:#86efac; }
    .chip-warn { background:#fef3c7; color:#92400e; border-color:#fcd34d; }

    .result-banner {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.35);
        margin: 1rem 0;
    }
    .result-banner h2 { color: white; margin: 0; font-size: 1.7rem; }
    .result-banner p  { color: rgba(255,255,255,0.9); margin: 0.3rem 0 0 0; }

    .top-card {
        background: #f8f9ff;
        border-left: 4px solid #667eea;
        padding: 0.7rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPERS
# =============================================================================
def _is_non_numeric(v):
    """Return True if value cannot be parsed as a number."""
    try:
        float(v)
        return False
    except (ValueError, TypeError):
        return True

# =============================================================================
# LOAD DATA & MODEL
# =============================================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model(path): return joblib.load(path)

@st.cache_data(show_spinner="Loading data...")
def load_data(path): return pd.read_csv(path)

MODEL_PATH = "final_proj.pkl"
DATA_PATH = "Responses.csv"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ `{MODEL_PATH}` not found."); st.stop()
if not os.path.exists(DATA_PATH):
    st.error(f"❌ `{DATA_PATH}` not found."); st.stop()

model = load_model(MODEL_PATH)
df = load_data(DATA_PATH)

df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df.columns = [c.strip() for c in df.columns]
TARGET = "Choose your track"
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

# =============================================================================
# LABEL MAPPING
# =============================================================================
label_mapping = {i: name for i, name in enumerate(sorted(df[TARGET].unique()))}

# =============================================================================
# FEATURE COLUMNS
# =============================================================================
feature_cols = [c for c in df.columns if c != TARGET]

# =============================================================================
# EXTRACT MODEL FEATURE NAMES
# =============================================================================
def get_model_feature_names(pipeline):
    try:
        if hasattr(pipeline, "feature_names_in_"):
            return list(pipeline.feature_names_in_)
        first = pipeline.steps[0][1]
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
    except Exception:
        pass
    return None

model_features = get_model_feature_names(model) or feature_cols

# =============================================================================
# Q-NAME → REAL CSV COLUMN MAPPING
# =============================================================================
def build_feature_mapping(model_feats, df_columns, target_col):
    df_feature_cols = [c for c in df_columns if c != target_col]
    uses_q = all(
        str(f).strip().upper().startswith("Q") and str(f).strip()[1:].isdigit()
        for f in model_feats
    )
    if uses_q and len(model_feats) == len(df_feature_cols):
        sorted_feats = sorted(model_feats, key=lambda x: int(str(x).strip()[1:]))
        return dict(zip(sorted_feats, df_feature_cols)), sorted_feats
    return {f: f for f in model_feats}, list(model_feats)

FEATURE_MAP, MODEL_FEATURE_ORDER = build_feature_mapping(
    model_features, df.columns.tolist(), TARGET
)

# =============================================================================
# SHORT LABELS
# =============================================================================
SHORT_LABELS = {
    "Q1":  "Programming Expertise",
    "Q2":  "Coding Frequency",
    "Q3":  "Follow Tech News",
    "Q4":  "Willing to Learn Code",
    "Q5":  "Enjoy Visual Design",
    "Q6":  "Analyze User Behavior",
    "Q7":  "Strong Math Background",
    "Q8":  "Enjoy Data Problems",
    "Q9":  "Interest in Data Research",
    "Q10": "Passion for Data Insights",
    "Q11": "Interest in Algorithms/ML",
    "Q12": "Server-side Programming",
    "Q13": "Build Innovative Software",
    "Q14": "Mobile Apps Interest",
    "Q15": "Digital Security Concern",
    "Q16": "Enjoy Encryption/Firewalls",
    "Q17": "Creative Problem Solving",
    "Q18": "Responsive Web Design",
}

# =============================================================================
# ORDINAL ORDERINGS
# =============================================================================
ORDINAL_ORDERS = {
    "Q1": ["Beginner", "Intermediate", "Expert"],
    "Q2": ["Never", "Rarely", "Monthly", "Weekly", "Daily"],
    "Q3": ["Never", "Rarely", "Monthly", "Weekly", "Daily"],
}

def get_ordered_options(q_name, real_col, dataframe):
    raw = dataframe[real_col].dropna().astype(str).unique().tolist()
    if q_name in ORDINAL_ORDERS:
        desired = ORDINAL_ORDERS[q_name]
        lookup = {v.lower(): v for v in raw}
        ordered = []
        for want in desired:
            if want.lower() in lookup:
                ordered.append(lookup[want.lower()])
        for v in raw:
            if v not in ordered:
                ordered.append(v)
        return ordered
    return sorted(raw)

# =============================================================================
# ENCODERS
# =============================================================================
@st.cache_data(show_spinner=False)
def build_encoders(dataframe, feature_map):
    encoders = {}
    for q_name, real_col in feature_map.items():
        if real_col in dataframe.columns:
            uniq = sorted(dataframe[real_col].dropna().astype(str).unique().tolist())
            encoders[q_name] = {v: i for i, v in enumerate(uniq)}
    return encoders

ENCODERS = build_encoders(df, FEATURE_MAP)

# =============================================================================
# BUILD MODEL INPUT
# =============================================================================
def build_model_input(user_input: dict) -> pd.DataFrame:
    row = {}
    for q_name in MODEL_FEATURE_ORDER:
        ans = user_input.get(q_name)
        if ans is None:
            real_col = FEATURE_MAP[q_name]
            ans = df[real_col].mode()[0]
        enc = ENCODERS.get(q_name, {}).get(str(ans), 0)
        row[q_name] = enc
    return pd.DataFrame([row], columns=MODEL_FEATURE_ORDER)

# =============================================================================
# NAVIGATION — session-state driven with programmatic jump support
# =============================================================================
PAGES = ["🏠 Home", "📊 Distributions", "🔍 Model Insights", "🎯 Predict Track"]

if "current_page" not in st.session_state:
    st.session_state.current_page = PAGES[0]

# Handle a programmatic navigation request BEFORE the radio widget is created
if st.session_state.get("nav_target"):
    target = st.session_state.nav_target
    st.session_state.current_page = target
    st.session_state["nav_radio"] = target      # force widget to reflect new value
    st.session_state.nav_target = None          # consume request

if st.session_state.current_page not in PAGES:
    st.session_state.current_page = PAGES[0]

st.sidebar.markdown(
    """
    <div style='text-align:center; padding: 0.8rem 0 0.4rem 0;'>
        <div style='font-size: 2.3rem;'>🚀</div>
        <div style='font-size: 1.25rem; font-weight: 800; color: white;'>Tech Track</div>
        <div style='font-size: 0.8rem; color: rgba(255,255,255,0.7);'>Recommender System</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    PAGES,
    index=PAGES.index(st.session_state.current_page),
    key="nav_radio",
    label_visibility="collapsed",
)

st.session_state.current_page = page

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style='color: rgba(255,255,255,0.75); font-size: 0.82rem; padding: 0.5rem; line-height:1.7;'>
        <b>Model:</b> Random Forest<br>
        <b>Features:</b> {len(MODEL_FEATURE_ORDER)}<br>
        <b>Classes:</b> {df[TARGET].nunique()}<br>
        <b>Responses:</b> {len(df):,}
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# HOME
# =============================================================================
if page == "🏠 Home":
    st.title("🚀 Tech Track Recommender")
    st.markdown(
        "<p style='font-size:1.1rem;color:#555;'>"
        "Discover your ideal tech career path with a short interactive quiz, powered by "
        "machine learning trained on real survey data.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Responses", f"{len(df):,}")
    c2.metric("🎯 Tracks", df[TARGET].nunique())
    c3.metric("❓ Questions", len(feature_cols))
    c4.metric("🏆 Top Track", df[TARGET].value_counts().idxmax())

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📌 Track Distribution")
        tc = df[TARGET].value_counts().reset_index()
        tc.columns = ["Track", "Count"]
        chart = (
            alt.Chart(tc)
            .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8)
            .encode(
                x=alt.X("Count:Q", title="Responses"),
                y=alt.Y("Track:N", sort="-x", title=""),
                color=alt.Color("Track:N", scale=alt.Scale(scheme="viridis"), legend=None),
                tooltip=["Track", "Count"],
            )
            .properties(height=380)
        )
        st.altair_chart(chart, use_container_width=True)

    with col_right:
        st.subheader("✨ Get Started")
        st.markdown(
            """
            1. **📊 Explore** distributions
            2. **🔍 Inspect** the model's drivers  
            3. **🎯 Take** the 1-minute quiz
            4. **🚀 Get** your recommendation
            """
        )
        if st.button("🎯 Start Quiz →", use_container_width=True, type="primary"):
            st.session_state.nav_target = "🎯 Predict Track"
            st.rerun()

        st.markdown("---")
        st.subheader("🛠️ Tech Stack")
        st.markdown(
            "`Streamlit` · `scikit-learn` · `pandas` · `Altair` · `joblib`"
        )

# =============================================================================
# DISTRIBUTIONS
# =============================================================================
elif page == "📊 Distributions":
    st.title("📊 Distributions")
    st.markdown("Explore how survey answers are distributed across the dataset.")

    with st.expander("🔎 Filters", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            selected_tracks = st.multiselect(
                "Track",
                options=sorted(df[TARGET].unique()),
                default=sorted(df[TARGET].unique()),
            )
        with c2:
            exp_col = "1- What is your current level expertise in computer programming?"
            if exp_col in df.columns:
                selected_exp = st.multiselect(
                    "Expertise",
                    options=sorted(df[exp_col].dropna().unique()),
                    default=sorted(df[exp_col].dropna().unique()),
                )
            else:
                selected_exp = None

    filtered = df[df[TARGET].isin(selected_tracks)]
    if selected_exp is not None and exp_col in df.columns:
        filtered = filtered[filtered[exp_col].isin(selected_exp)]

    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} responses**")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📊 Categorical Answer Distribution")

        cat_cols = []
        for col in feature_cols:
            if col not in filtered.columns:
                continue
            dtype = filtered[col].dtype
            is_cat = False

            if dtype == "object" or dtype.name in ("category", "string"):
                is_cat = True
            else:
                try:
                    n_unique = filtered[col].nunique(dropna=True)
                    if 0 < n_unique <= 30:
                        sample = filtered[col].dropna().astype(str).head(5).tolist()
                        if any(_is_non_numeric(v) for v in sample):
                            is_cat = True
                except Exception:
                    pass

            if is_cat:
                cat_cols.append(col)

        if len(filtered) == 0:
            st.warning("⚠️ No rows match the current filters.")
        elif not cat_cols:
            st.info("ℹ️ No categorical columns available in the filtered data.")
            with st.expander("🔍 Debug — column dtypes"):
                st.dataframe(
                    pd.DataFrame({
                        "Column": list(filtered.columns),
                        "Dtype": [str(d) for d in filtered.dtypes],
                        "Unique": [filtered[c].nunique() for c in filtered.columns],
                        "Sample": [
                            str(filtered[c].dropna().head(2).tolist())
                            for c in filtered.columns
                        ],
                    }),
                    use_container_width=True,
                )
        else:
            sel = st.selectbox("Question", options=cat_cols, key="dist_cat")

            if sel is None or pd.isna(sel):
                st.warning("⚠️ Please select a question.")
            else:
                vc = filtered[sel].dropna().value_counts().reset_index()
                vc.columns = ["Answer", "Count"]

                if len(vc) == 0:
                    st.info(f"ℹ️ No values available for **{sel}**.")
                else:
                    ch = (
                        alt.Chart(vc)
                        .mark_arc(innerRadius=60)
                        .encode(
                            theta=alt.Theta("Count:Q"),
                            color=alt.Color(
                                "Answer:N",
                                scale=alt.Scale(scheme="set2"),
                                sort=alt.EncodingSortField(
                                    field="Count", order="descending"
                                ),
                            ),
                            tooltip=["Answer", "Count"],
                        )
                        .properties(height=380)
                    )
                    st.altair_chart(ch, use_container_width=True)

    with c2:
        st.subheader("🎯 Track Distribution")
        if len(filtered) == 0:
            st.warning("⚠️ No data to display.")
        else:
            tc = filtered[TARGET].value_counts().reset_index()
            tc.columns = ["Track", "Count"]
            ch2 = (
                alt.Chart(tc)
                .mark_bar(cornerRadiusTopRight=8)
                .encode(
                    x=alt.X("Count:Q"),
                    y=alt.Y("Track:N", sort="-x"),
                    color=alt.Color(
                        "Track:N",
                        scale=alt.Scale(scheme="tableau10"),
                        legend=None,
                    ),
                    tooltip=["Track", "Count"],
                )
                .properties(height=380)
            )
            st.altair_chart(ch2, use_container_width=True)

# =============================================================================
# MODEL INSIGHTS
# =============================================================================
elif page == "🔍 Model Insights":
    st.title("🔍 Model Insights")

    rf_model = None
    if isinstance(model, Pipeline):
        for _, step in model.steps:
            if isinstance(step, RandomForestClassifier):
                rf_model = step
                break
    elif isinstance(model, RandomForestClassifier):
        rf_model = model

    if rf_model is None:
        st.warning("⚠️ Could not find RandomForestClassifier in model.")
        st.stop()

    importances = rf_model.feature_importances_
    feature_names = (
        MODEL_FEATURE_ORDER if len(importances) == len(MODEL_FEATURE_ORDER)
        else [f"Q{i}" for i in range(len(importances))]
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Question": [SHORT_LABELS.get(q, q) for q in feature_names],
        "Importance": importances,
    }).sort_values("Importance", ascending=False)

    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("📊 Feature Importances")
        top_n = st.slider("Show top N features", 5, min(30, len(importance_df)), 12)
        top_df = importance_df.head(top_n).copy()

        chart = (
            alt.Chart(top_df)
            .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8)
            .encode(
                x=alt.X("Importance:Q", title="Importance"),
                y=alt.Y("Question:N", sort="-x", title=""),
                color=alt.Color(
                    "Importance:Q",
                    scale=alt.Scale(scheme="blues"),
                    legend=None,
                ),
                tooltip=[
                    "Feature", "Question",
                    alt.Tooltip("Importance:Q", format=".4f"),
                ],
            )
            .properties(height=max(400, top_n * 32))
        )
        st.altair_chart(chart, use_container_width=True)

    with c2:
        st.subheader("🌲 Model Details")
        st.markdown(f"""
        - **Algorithm:** Random Forest
        - **Trees:** `{rf_model.n_estimators}`
        - **Max Depth:** `{rf_model.max_depth or "Unlimited"}`
        - **Features:** `{rf_model.n_features_in_}`
        - **Classes:** `{rf_model.n_classes_}`
        """)
        st.subheader("🏷️ Target Classes")
        # ✅ Show only the class names (no index numbers)
        for cls in rf_model.classes_:
            name = label_mapping.get(cls, cls) if isinstance(cls, (int, np.integer)) else cls
            st.markdown(f"- **{name}**")

    st.markdown("---")
    st.subheader("📋 Full Feature Importance Table")
    st.dataframe(
        importance_df[["Feature", "Question", "Importance"]],
        use_container_width=True, height=400,
    )

# =============================================================================
# PREDICT TRACK
# =============================================================================
elif page == "🎯 Predict Track":
    st.title("🎯 Predict Your Tech Track")
    st.markdown(
        "<p style='color:#555;'>Answer the short quiz and click <b>Get Recommendation</b>."
        " <b>All questions must be answered.</b></p>",
        unsafe_allow_html=True,
    )

    YESNO_QS = ["Q4","Q5","Q6","Q7","Q8","Q9","Q10","Q11",
                "Q12","Q13","Q14","Q15","Q16","Q17","Q18"]
    DROPDOWN_QS = ["Q1","Q2","Q3"]

    user_input = {}

    with st.form("compact_quiz", clear_on_submit=False):
        st.markdown("<div class='q-section'>🎓 Background</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for i, q in enumerate(DROPDOWN_QS):
            real_col = FEATURE_MAP[q]
            label = SHORT_LABELS.get(q, real_col)
            options = get_ordered_options(q, real_col, df)
            col = [c1, c2, c3][i]
            with col:
                user_input[q] = st.selectbox(
                    f"**{q}** · {label}",
                    options=options,
                    key=f"in_{q}",
                )

        st.markdown(
            "<div class='q-section'>💡 Interests & Skills</div>",
            unsafe_allow_html=True,
        )

        rows = [YESNO_QS[i:i+2] for i in range(0, len(YESNO_QS), 2)]
        for row_qs in rows:
            cols = st.columns(2)
            for i, q in enumerate(row_qs):
                real_col = FEATURE_MAP[q]
                label = SHORT_LABELS.get(q, real_col)
                options = get_ordered_options(q, real_col, df)
                is_yesno = set(o.lower() for o in options) <= {"yes", "no"}

                with cols[i]:
                    if is_yesno:
                        user_input[q] = st.radio(
                            f"**{q}** · {label}",
                            options=["Yes", "No"],
                            index=None,
                            horizontal=True,
                            key=f"in_{q}",
                        )
                    else:
                        user_input[q] = st.selectbox(
                            f"**{q}** · {label}",
                            options=options,
                            key=f"in_{q}",
                        )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🚀 Get Recommendation", use_container_width=True
        )

    answered_yesno = sum(1 for q in YESNO_QS if user_input.get(q) in ("Yes", "No"))
    total_yesno = len(YESNO_QS)
    answered_all = (
        answered_yesno == total_yesno
        and all(user_input.get(q) is not None for q in DROPDOWN_QS)
    )

    if answered_all:
        st.markdown(
            f"<div style='margin-top:0.8rem;'>"
            f"<span class='chip chip-yes'>✅ All {total_yesno + len(DROPDOWN_QS)} questions answered</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='margin-top:0.8rem;'>"
            f"<span class='chip chip-warn'>⚠️ {answered_yesno}/{total_yesno} interest questions answered</span>"
            f"<span class='chip'>📊 {len(MODEL_FEATURE_ORDER)} total questions</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if submitted:
        unanswered = [q for q in YESNO_QS if user_input.get(q) not in ("Yes", "No")]

        if unanswered:
            st.error(
                f"❌ Please answer all questions. "
                f"Missing {len(unanswered)}: "
                + ", ".join(unanswered[:8])
                + ("..." if len(unanswered) > 8 else "")
            )
        else:
            try:
                model_input = build_model_input(user_input)

                prediction = model.predict(model_input)[0]
                probabilities = model.predict_proba(model_input)[0]
                classes = model.classes_

                pred_name = (
                    label_mapping.get(prediction, prediction)
                    if isinstance(prediction, (int, np.integer)) else prediction
                )

                decoded = []
                for c in classes:
                    decoded.append(
                        label_mapping.get(c, c)
                        if isinstance(c, (int, np.integer)) else c
                    )

                prob_df = pd.DataFrame({
                    "Track": decoded,
                    "Probability": probabilities,
                }).sort_values("Probability", ascending=False)

                st.markdown(
                    f"""
                    <div class='result-banner'>
                        <h2>🎉 Recommended Track: {pred_name}</h2>
                        <p>Confidence: {prob_df.iloc[0]['Probability']:.1%}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c1, c2 = st.columns([3, 2])

                with c1:
                    st.subheader("📊 Confidence Across All Tracks")
                    chart = (
                        alt.Chart(prob_df)
                        .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8)
                        .encode(
                            x=alt.X(
                                "Probability:Q",
                                axis=alt.Axis(format="%"),
                                title="Probability",
                            ),
                            y=alt.Y("Track:N", sort="-x", title=""),
                            color=alt.Color(
                                "Probability:Q",
                                scale=alt.Scale(scheme="tealblues"),
                                legend=None,
                            ),
                            tooltip=["Track", alt.Tooltip("Probability:Q", format=".2%")],
                        )
                        .properties(height=380)
                    )
                    st.altair_chart(chart, use_container_width=True)

                with c2:
                    st.subheader("🏆 Top 3 Matches")
                    for i, row in prob_df.head(3).iterrows():
                        idx = list(prob_df.head(3).index).index(i)
                        icon = ["🥇", "🥈", "🥉"][idx]
                        st.markdown(
                            f"""
                            <div class='top-card'>
                                <b>{icon} {row['Track']}</b><br>
                                <span style='color:#666;font-size:0.9rem;'>
                                    Probability: {row['Probability']:.1%}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with st.expander("🔍 Review Your Answers", expanded=False):
                    recap = pd.DataFrame({
                        "Q": MODEL_FEATURE_ORDER,
                        "Question": [SHORT_LABELS.get(q, q) for q in MODEL_FEATURE_ORDER],
                        "Your Answer": [user_input.get(q, "—") for q in MODEL_FEATURE_ORDER],
                    })
                    st.dataframe(recap, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                with st.expander("🛠️ Traceback"):
                    st.exception(e)

# =============================================================================
# FOOTER
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align:center;color:rgba(255,255,255,0.6);"
    "font-size:0.78rem;'>Built with ❤️ using Streamlit</div>",
    unsafe_allow_html=True,
)