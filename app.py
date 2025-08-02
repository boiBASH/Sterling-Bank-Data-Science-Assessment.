import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import mlflow
from PIL import Image

# ---------- CONFIG ----------
TARGET = "Default_status"
LEAK_COLS = ["DAYS_TO_MATURITY", "CONTRACT_MAT_DATE", "report_date", "PayinAccount_Last_LOD_Date"]
MLFLOW_TRACKING_URI = "https://dagshub.com/boiBASH/Sterling-Bank-Data-Science-Assessment..mlflow"
RF_RUN_ID = "ab07579390ea427eb320b944b63c8f66"
RF_MODEL_NAME = "RandomForest_SMOTE_Optimized"

# ---------- MLflow setup ----------
def setup_mlflow():
    os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["dagshub"]["username"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["dagshub"]["token"]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@st.cache_resource
def load_model():
    setup_mlflow()
    uri = f"runs:/{RF_RUN_ID}/{RF_MODEL_NAME}"
    return mlflow.sklearn.load_model(uri)

model = load_model()

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Sterling Loan Explorer", layout="wide", initial_sidebar_state="expanded")
# Custom CSS for subtle visual polish
st.markdown(
    """
    <style>
    .stApp { background: #f5f7fa; }
    .metric-label { font-size: 0.9rem; }
    .card { padding: 1rem; border-radius: 12px; background: white; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- HEADER WITH LOGO ----------
col_logo, col_title = st.columns([1, 9])
with col_logo:
    try:
        logo = Image.open("sterling_bank_logo.png")  # adjust path if in assets/
        st.image(logo, width=80)
    except FileNotFoundError:
        st.text("Sterling Bank")
with col_title:
    st.markdown("<h1 style='margin-bottom:4px;'>📊 Sterling Loan Data Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin-top:0;color:gray;'>Clean data visualization + default risk scoring powered by your MLflow model.</p>", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
@st.cache_data
def load_excel(path):
    return pd.read_excel(path)

uploaded = st.file_uploader("Upload cleaned loan Excel file", type=["xlsx"], help="If present in repo, it'll auto-load.")
if uploaded:
    df = load_excel(uploaded)
else:
    try:
        df = load_excel("cleaned_loan_data.xlsx")
    except FileNotFoundError:
        st.error("No dataset found. Upload cleaned_loan_data.xlsx to proceed.")
        st.stop()

# ---------- SIDEBAR FILTERS ----------
st.sidebar.header("Filters & Cohorts")
with st.sidebar.expander("Segment Selection", expanded=True):
    sectors = st.multiselect("Sector", options=sorted(df["sector"].dropna().unique()))
    facility = st.multiselect("Facility Type", options=sorted(df["FACILITY_TYPE"].dropna().unique()))
    is_active = st.selectbox("Is Active Loans", ["All", "Active", "Inactive"], index=0)
    employment = st.multiselect("Employment Status", options=sorted(df["employment_status"].dropna().unique()))
    default_kind = st.multiselect("Default Status Kind", options=sorted(df["Default_status_kind"].dropna().unique()))
    loan_age_range = st.slider(
        "Loan Age Days",
        min_value=int(df["loan_age_days"].dropna().min()),
        max_value=int(df["loan_age_days"].dropna().quantile(0.99)),
        value=(0, int(df["loan_age_days"].dropna().quantile(0.75))),
        step=10,
    )

# ---------- APPLY FILTERS ----------
filtered = df.copy()
if sectors:
    filtered = filtered[filtered["sector"].isin(sectors)]
if facility:
    filtered = filtered[filtered["FACILITY_TYPE"].isin(facility)]
if is_active != "All":
    filtered = filtered[filtered["Is_Active_loans"].str.contains(is_active, case=False, na=False)]
if employment:
    filtered = filtered[filtered["employment_status"].isin(employment)]
if default_kind:
    filtered = filtered[filtered["Default_status_kind"].isin(default_kind)]
filtered = filtered[
    (filtered["loan_age_days"] >= loan_age_range[0]) & (filtered["loan_age_days"] <= loan_age_range[1])
]

# ---------- KEY METRICS ----------
st.markdown("## 🔑 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
default_rate = filtered[TARGET].mean() if TARGET in filtered.columns else 0
col1.metric("Total Loans", f"{len(filtered):,}")
col2.metric("Default Rate", f"{default_rate:.2%}")
col3.metric("Avg Loan Age (days)", f"{filtered['loan_age_days'].mean():.1f}" if "loan_age_days" in filtered.columns else "N/A")
col4.metric("Unique Sectors", filtered["sector"].nunique())

# ---------- DISTRIBUTIONS AND BREAKDOWNS ----------
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
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Default Status Kind")
        if "Default_status_kind" in filtered.columns:
            kind_df = (
                filtered["Default_status_kind"]
                .value_counts()
                .reset_index()
                .rename(columns={"index": "kind", "Default_status_kind": "count"})
            )
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
        st.markdown("### Time Trend")
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

# ---------- SEGMENT PERFORMANCE ----------
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

# ---------- NUMERIC EXPLORER ----------
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

# ---------- CORRELATION HEATMAP ----------
st.markdown("## 🔗 Correlation Matrix (Clustered)")
if numeric_cols:
    corr_df = filtered[numeric_cols].dropna()
    if corr_df.shape[0] > 1000:
        corr_df = corr_df.sample(1000, random_state=42)
    corr = corr_df.corr()
    link = linkage(corr, method="average")
    order = leaves_list(link)
    corr_ord = corr.iloc[order, order]
    fig_corr = go.Figure(
        data=go.Heatmap(
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

# ---------- COHORT SPOTLIGHT ----------
st.markdown("## 🧪 Cohort Spotlight")
c1, c2 = st.columns(2)
with c1:
    if "loan_age_days" in filtered.columns:
        loan_bins = st.slider(
            "Loan Age Cohort (days)", int(filtered["loan_age_days"].min()), int(filtered["loan_age_days"].max()), (0, 365)
        )
        sub = filtered[
            (filtered["loan_age_days"] >= loan_bins[0]) & (filtered["loan_age_days"] <= loan_bins[1])
        ]
        rate = sub[TARGET].mean() if not sub.empty else 0
        st.metric(f"Default rate for loan_age_days in {loan_bins}", f"{rate:.2%}")
with c2:
    if "customer_tenure_days" in filtered.columns:
        tenure_bins = st.slider(
            "Customer Tenure Cohort (days)",
            0,
            int(filtered["customer_tenure_days"].quantile(0.9)),
            (0, 365),
        )
        sub2 = filtered[
            (filtered["customer_tenure_days"] >= tenure_bins[0])
            & (filtered["customer_tenure_days"] <= tenure_bins[1])
        ]
        rate2 = sub2[TARGET].mean() if not sub2.empty else 0
        st.metric(f"Default rate for tenure in {tenure_bins}", f"{rate2:.2%}")

# ---------- TOP RISKY SEGMENTS ----------
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
    seg_df_display = seg_df.copy()
    seg_df_display["default_rate"] = seg_df_display["default_rate"].map("{:.1%}".format)
    st.dataframe(seg_df_display.head(20), use_container_width=True)

# ---------- MODEL LOADING & PREDICTION ----------
st.markdown("## 🧠 Default Risk Scoring")
st.caption("Model loaded from DagsHub MLflow (Random Forest). Provide input and get probability.")

# Setup MLflow credentials from secrets
def setup_mlflow():
    os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["dagshub"]["username"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["dagshub"]["token"]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@st.cache_resource
def load_model():
    setup_mlflow()
    uri = f"runs:/{RF_RUN_ID}/{RF_MODEL_NAME}"
    return mlflow.sklearn.load_model(uri)

model = load_model()

# Prediction threshold
threshold = st.slider("Probability threshold for default", 0.0, 1.0, 0.5, 0.01)

# Prepare template input (drop leaks and target)
base_example = filtered.copy()
for c in LEAK_COLS:
    if c in base_example.columns:
        base_example = base_example.drop(columns=[c])
if TARGET in base_example.columns:
    base_example = base_example.drop(columns=[TARGET])
example_row = base_example.head(1)

# Single scoring
with st.expander("Single loan scoring", expanded=True):
    st.markdown("Fill in values and score.")
    input_vals = {}
    with st.form("score_form"):
        for col in example_row.columns:
            if pd.api.types.is_numeric_dtype(example_row[col]):
                input_vals[col] = st.number_input(col, value=float(example_row[col].iloc[0]), format="%.4f")
            else:
                input_vals[col] = st.text_input(col, value=str(example_row[col].iloc[0]))
        submitted = st.form_submit_button("Score")
    if submitted:
        input_df = pd.DataFrame([input_vals])
        try:
            proba = model.predict_proba(input_df)[:, 1][0]
            label = int(proba >= threshold)
            st.success(f"Default probability: {proba:.3f} → Predicted label: {label}")
        except Exception as e:
            st.error(f"Scoring failed: {e}")

# Batch scoring
with st.expander("Batch scoring", expanded=False):
    st.markdown("Upload a CSV with the same features (excluding leak columns & target).")
    uploaded_batch = st.file_uploader("Upload CSV for scoring", type=["csv"], key="batch_score")
    if uploaded_batch:
        batch = pd.read_csv(uploaded_batch)
        # Drop leak & target if present
        batch = batch.drop(columns=[c for c in LEAK_COLS if c in batch.columns], errors="ignore")
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

# ---------- EXPORT ----------
st.markdown("## 📦 Export & Download")
st.download_button("Download filtered slice", filtered.to_csv(index=False).encode("utf-8"), "filtered_loans.csv", "text/csv")

# ---------- FOOTER ----------
st.markdown(
    """
    ---
    **Notes:**  
    • Filters dynamically drive all segment views.  
    • Prediction uses the remote Random Forest from MLflow; ensure input feature alignment.  
    • You can style further (dark theme / branding) by customizing CSS or Plotly templates.  
    """
)
