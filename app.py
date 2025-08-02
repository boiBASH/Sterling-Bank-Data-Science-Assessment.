import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
import joblib
from PIL import Image
import os
import traceback

# === CONFIG ===
TARGET = "Default_status"
LEAK_COLS = ["DAYS_TO_MATURITY", "CONTRACT_MAT_DATE", "report_date", "PayinAccount_Last_LOD_Date"]
MODEL_PATH = "model.pkl"  # your downloaded pipeline (could be full or light)
LOGO_PATH = "sterling bank logo.png"
DATA_PATH = "cleaned_loan_data.xlsx"

# === PAGE SETUP ===
st.set_page_config(page_title="Sterling Loan Explorer", layout="wide")
col_logo, col_title = st.columns([1, 8])
with col_logo:
    try:
        st.image(LOGO_PATH, width=80)
    except FileNotFoundError:
        st.markdown("**Sterling Bank**")
with col_title:
    st.markdown("<h1 style='margin:0;'>📊 Sterling Loan Data Explorer & Risk Scoring</h1>", unsafe_allow_html=True)
    st.markdown("Prediction-only dashboard using the provided model. Explore cohorts and score loans.", unsafe_allow_html=True)

# === LOAD DATA ===
@st.cache_data
def load_data(path):
    return pd.read_excel(path)

if not os.path.exists(DATA_PATH):
    st.error(f"Data file '{DATA_PATH}' not found in repo root. Commit it or upload manually.")
    st.stop()

df = load_data(DATA_PATH)

# === DEDUPE COLUMNS ===
def make_cols_unique(df):
    seen = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df

original_cols = df.columns.tolist()
df = make_cols_unique(df)
dupes = [c for c in original_cols if original_cols.count(c) > 1]
if dupes:
    st.warning(f"Duplicate column names were renamed: {set(dupes)}")

# === SIDEBAR FILTERS ===
st.sidebar.header("Filters & Cohorts")
sectors = st.sidebar.multiselect("Sector", options=sorted(df["sector"].dropna().unique()))
facilities = st.sidebar.multiselect("Facility Type", options=sorted(df["FACILITY_TYPE"].dropna().unique()))
is_active = st.sidebar.selectbox("Is Active Loans", ["All", "Active", "Inactive"])
employment = st.sidebar.multiselect("Employment Status", options=sorted(df["employment_status"].dropna().unique()))
default_kind = st.sidebar.multiselect("Default Status Kind", options=sorted(df["Default_status_kind"].dropna().unique()))
loan_age_range = st.sidebar.slider(
    "Loan Age Days",
    min_value=int(df["loan_age_days"].dropna().min()),
    max_value=int(df["loan_age_days"].dropna().quantile(0.99)),
    value=(0, int(df["loan_age_days"].dropna().quantile(0.75))),
    step=10,
)

# === APPLY FILTERS ===
filtered = df.copy()
if sectors:
    filtered = filtered[filtered["sector"].isin(sectors)]
if facilities:
    filtered = filtered[filtered["FACILITY_TYPE"].isin(facilities)]
if is_active != "All":
    filtered = filtered[filtered["Is_Active_loans"].str.contains(is_active, case=False, na=False)]
if employment:
    filtered = filtered[filtered["employment_status"].isin(employment)]
if default_kind:
    filtered = filtered[filtered["Default_status_kind"].isin(default_kind)]
filtered = filtered[
    (filtered["loan_age_days"] >= loan_age_range[0]) & (filtered["loan_age_days"] <= loan_age_range[1])
]

# === KEY METRICS ===
st.subheader("🔑 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
default_rate = filtered[TARGET].mean() if TARGET in filtered.columns else 0
col1.metric("Total Loans", f"{len(filtered):,}")
col2.metric("Default Rate", f"{default_rate:.2%}")
col3.metric("Avg Loan Age (days)", f"{filtered['loan_age_days'].mean():.1f}" if "loan_age_days" in filtered.columns else "N/A")
col4.metric("Unique Sectors", filtered["sector"].nunique())

# === OVERVIEW ===
st.markdown("## 📈 Overview & Breakdown")
with st.container():
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### Default Status")
        if TARGET in filtered.columns:
            fig = px.pie(
                filtered,
                names=TARGET,
                title="Defaults vs Non-defaults",
                hole=0.35,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Default Status Kind")
        if "Default_status_kind" in filtered.columns:
            kind_df = filtered["Default_status_kind"].value_counts().reset_index()
            kind_df.columns = ["kind", "count"]
            fig_kind = px.bar(
                kind_df,
                x="kind",
                y="count",
                title="Default Status Kind",
                color="count",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_kind, use_container_width=True)
    with c2:
        st.markdown("### Default Rate Over Time")
        if "report_date" in filtered.columns:
            time_series = (
                filtered.assign(report_date=pd.to_datetime(filtered["report_date"]).dt.date)
                .groupby("report_date")[TARGET]
                .mean()
                .reset_index(name="default_rate")
            )
            fig_time = px.line(
                time_series,
                x="report_date",
                y="default_rate",
                title="Default Rate Over Time",
                markers=True,
            )
            fig_time.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_time, use_container_width=True)

# === SEGMENT PERFORMANCE ===
st.markdown("## 🧩 Segment Performance")
seg1, seg2 = st.columns(2)
with seg1:
    if "sector" in filtered.columns:
        sector_perf = (
            filtered.groupby("sector")
            .agg(default_rate=(TARGET, "mean"), count=(TARGET, "size"))
            .reset_index()
            .sort_values("default_rate", ascending=False)
        )
        fig_sector = px.bar(
            sector_perf,
            x="sector",
            y="default_rate",
            color="count",
            title="Default Rate by Sector",
            hover_data={"count": True, "default_rate": ":.1%"},
        )
        fig_sector.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_sector, use_container_width=True)
with seg2:
    if "FACILITY_TYPE" in filtered.columns:
        fac_perf = (
            filtered.groupby("FACILITY_TYPE")
            .agg(default_rate=(TARGET, "mean"), count=(TARGET, "size"))
            .reset_index()
            .sort_values("default_rate", ascending=False)
        )
        fig_fac = px.bar(
            fac_perf.head(15),
            x="FACILITY_TYPE",
            y="default_rate",
            title="Top Facility Types by Default Rate",
            hover_data={"count": True},
        )
        fig_fac.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_fac, use_container_width=True)

# === NUMERIC EXPLORER ===
st.markdown("## 🔬 Numeric Feature Explorer")
numeric_cols = filtered.select_dtypes(include=["number"]).columns.tolist()
if numeric_cols:
    chosen = st.selectbox("Select numeric feature", options=numeric_cols, index=0)
    d1, d2 = st.columns(2)
    with d1:
        fig_hist = px.histogram(
            filtered,
            x=chosen,
            nbins=50,
            title=f"Distribution of {chosen}",
            marginal="box",
            template="plotly_white",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    with d2:
        if TARGET in filtered.columns:
            fig_violin = px.violin(
                filtered,
                x=TARGET,
                y=chosen,
                box=True,
                points="all",
                title=f"{chosen} by Default Status",
                color=TARGET,
                color_discrete_map={0: "#00CC96", 1: "#EF553B"},
            )
            st.plotly_chart(fig_violin, use_container_width=True)

# === CORRELATION HEATMAP ===
st.markdown("## 🔗 Clustered Correlation Matrix")
if numeric_cols:
    corr_df = filtered[numeric_cols].dropna()
    if corr_df.shape[0] > 1000:
        corr_df = corr_df.sample(1000, random_state=42)
    corr = corr_df.corr()
    link = linkage(corr, method="average")
    order = leaves_list(link)
    corr_ord = corr.iloc[order, order]
    fig_corr = go.Figure(
        go.Heatmap(
            z=corr_ord.values,
            x=corr_ord.columns,
            y=corr_ord.index,
            colorscale="RdBu",
            zmid=0,
            hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
        )
    )
    fig_corr.update_layout(title="Clustered Correlation", height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

# === TOP RISKY SEGMENTS ===
st.markdown("## 🔎 Top Risky Segments")
segment_dims = ["sector", "FACILITY_TYPE", "employment_status"]
combo = st.multiselect("Segment dimensions", options=segment_dims, default=segment_dims[:2])
if combo:
    seg_df = (
        filtered.groupby(combo)
        .agg(default_rate=(TARGET, "mean"), count=(TARGET, "size"))
        .reset_index()
        .sort_values("default_rate", ascending=False)
    )
    seg_df["default_rate"] = seg_df["default_rate"].map("{:.1%}".format)
    st.dataframe(seg_df.head(15), use_container_width=True)

# === MODEL LOADING & PREDICTION ===
st.markdown("## 🧠 Default Risk Scoring")
st.caption("Uses local model. If it includes imblearn and errors surface, either install 'imbalanced-learn==0.11.0' or re-extract a light version.")

@st.cache_resource
def load_light_model(path):
    return joblib.load(path)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Place the .pkl in repo root.")
    st.stop()

model = None
load_error = None
try:
    model = load_light_model(MODEL_PATH)
except Exception as e:
    load_error = e
    st.error("Failed to load model. See guidance below.")
    st.markdown("**Likely causes / remedies:**")
    st.markdown(
        """
- The `.pkl` you provided contains `imblearn`/SMOTE, but `imbalanced-learn` isn't installed in this environment.  
  **Fix:** add to `requirements.txt` and reinstall:  
  `imbalanced-learn==0.11.0` (with compatible `scikit-learn==1.6.1`)  
- Or re-extract a light pipeline that omits SMOTE (no imblearn) using the extraction script in a Python 3.11 environment.  
"""
    )
    st.exception(load_error)
    st.stop()

threshold = st.slider("Default probability threshold", 0.0, 1.0, 0.5, 0.01)

# Single scoring
st.markdown("### Single Loan Scoring")
example = filtered.copy()
for c in LEAK_COLS:
    if c in example.columns:
        example = example.drop(columns=[c])
if TARGET in example.columns:
    example = example.drop(columns=[TARGET])
if not example.empty:
    input_vals = {}
    with st.form("single_score"):
        for col in example.columns:
            if pd.api.types.is_numeric_dtype(example[col]):
                input_vals[col] = st.number_input(col, value=float(example[col].iloc[0]), key=f"num_{col}")
            else:
                input_vals[col] = st.text_input(col, value=str(example[col].iloc[0]), key=f"txt_{col}")
        submitted = st.form_submit_button("Score Loan")
    if submitted:
        input_df = pd.DataFrame([input_vals])
        try:
            prob = model.predict_proba(input_df)[:, 1][0]
            label = int(prob >= threshold)
            st.success(f"Default probability: {prob:.3f} → Predicted label: {label}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.text(traceback.format_exc())

# Batch scoring
st.markdown("### Batch Scoring")
uploaded_batch = st.file_uploader("Upload CSV for batch scoring", type=["csv"], key="batch")
if uploaded_batch:
    batch = pd.read_csv(uploaded_batch)
    for c in LEAK_COLS:
        if c in batch.columns:
            batch = batch.drop(columns=[c])
    if TARGET in batch.columns:
        batch = batch.drop(columns=[TARGET])
    try:
        probs = model.predict_proba(batch)[:, 1]
        batch["default_probability"] = probs
        batch["predicted_default"] = (probs >= threshold).astype(int)
        st.dataframe(batch.head(50))
        st.download_button(
            "Download scored batch",
            batch.to_csv(index=False).encode("utf-8"),
            "scored_loans.csv",
            "text/csv",
        )
    except Exception as e:
        st.error(f"Batch scoring failed: {e}")
        st.text(traceback.format_exc())

# === EXPORT ===
st.markdown("## 📦 Export Filtered Slice")
st.download_button("Download filtered data", filtered.to_csv(index=False).encode("utf-8"), "filtered_loans.csv", "text/csv")

# === FOOTER ===
st.markdown(
    """
---
**Notes:**  
- If the provided `.pkl` has imblearn/SMOTE inside, add to your `requirements.txt`:  
  `scikit-learn==1.6.1` and `imbalanced-learn==0.11.0` so it loads.  
- For a lighter embeddable model, re-extract a pipeline without SMOTE (see extraction script).  
"""
)
