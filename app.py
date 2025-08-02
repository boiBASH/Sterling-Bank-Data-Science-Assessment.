import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.preprocessing import LabelEncoder

# === CONFIG ===
TARGET = "Default_status"
DROP_REDUNDANT = ["Default_status_kind", "ARR_STATUS", "LINE_DESC"]
LEAK_COLS = ["DAYS_TO_MATURITY", "CONTRACT_MAT_DATE", "report_date", "PayinAccount_Last_LOD_Date"]
MODEL_PATH = "light_rf_model.pkl"  # should be pipeline with fitted preprocessing + RF
DATA_PATH = "cleaned_loan_data.xlsx"
LOGO_PATH = "sterling bank logo.png"

st.set_page_config(page_title="Sterling Loan Explorer", layout="wide")

# === HEADER ===
col_logo, col_title = st.columns([1, 8])
with col_logo:
    try:
        st.image(LOGO_PATH, width=80)
    except FileNotFoundError:
        st.markdown("**Sterling Bank**")
with col_title:
    st.markdown("<h1 style='margin:0;'> Sterling Loan Explorer & Default Risk Scoring</h1>", unsafe_allow_html=True)
    st.markdown("Explore and score loans. Prediction is isolated in its own section.", unsafe_allow_html=True)

# === LOAD DATA ===
@st.cache_data
def load_raw_data(path):
    return pd.read_excel(path)

if not os.path.exists(DATA_PATH):
    st.error(f"{DATA_PATH} missing in repo root.")
    st.stop()

df_raw = load_raw_data(DATA_PATH)

# === PREPROCESS: label encode exactly like training ===
categorical_cols = df_raw.select_dtypes(include="object").columns.tolist()
for rm in DROP_REDUNDANT:
    if rm in categorical_cols:
        categorical_cols.remove(rm)
if TARGET in categorical_cols:
    categorical_cols.remove(TARGET)

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_raw[col] = df_raw[col].astype(str).fillna("___MISSING___")
    le.fit(df_raw[col])
    label_encoders[col] = le

def encode_dataframe(df_in):
    df = df_in.copy()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("___MISSING___")
            df[col] = label_encoders[col].transform(df[col])
    for c in DROP_REDUNDANT:
        if c in df.columns:
            df = df.drop(columns=c)
    return df

df_encoded = encode_dataframe(df_raw)

# === DEDUPE COLUMNS ===
def make_cols_unique(df):
    seen = {}
    new = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            new.append(c)
    df.columns = new
    return df

df_encoded = make_cols_unique(df_encoded)

# === SIDEBAR FILTERS ===
st.sidebar.header("Filters & Cohorts")
sectors = st.sidebar.multiselect("Sector", options=sorted(df_raw["sector"].dropna().unique()) if "sector" in df_raw.columns else [])
facilities = st.sidebar.multiselect("Facility Type", options=sorted(df_raw["FACILITY_TYPE"].dropna().unique()) if "FACILITY_TYPE" in df_raw.columns else [])
employment = st.sidebar.multiselect("Employment Status", options=sorted(df_raw["employment_status"].dropna().unique()) if "employment_status" in df_raw.columns else [])
default_kind = st.sidebar.multiselect("Default Status Kind", options=sorted(df_raw["Default_status_kind"].dropna().unique()) if "Default_status_kind" in df_raw.columns else [])
loan_age_range = st.sidebar.slider(
    "Loan Age Days",
    min_value=int(df_raw["loan_age_days"].dropna().min()) if "loan_age_days" in df_raw.columns else 0,
    max_value=int(df_raw["loan_age_days"].dropna().quantile(0.99)) if "loan_age_days" in df_raw.columns else 1,
    value=(0, int(df_raw["loan_age_days"].dropna().quantile(0.75))) if "loan_age_days" in df_raw.columns else (0, 1),
    step=10,
)

# Apply filters
filtered_raw = df_raw.copy()
if sectors and "sector" in filtered_raw.columns:
    filtered_raw = filtered_raw[filtered_raw["sector"].isin(sectors)]
if facilities and "FACILITY_TYPE" in filtered_raw.columns:
    filtered_raw = filtered_raw[filtered_raw["FACILITY_TYPE"].isin(facilities)]
if employment and "employment_status" in filtered_raw.columns:
    filtered_raw = filtered_raw[filtered_raw["employment_status"].isin(employment)]
if default_kind and "Default_status_kind" in filtered_raw.columns:
    filtered_raw = filtered_raw[filtered_raw["Default_status_kind"].isin(default_kind)]
if "loan_age_days" in filtered_raw.columns:
    filtered_raw = filtered_raw[
        (filtered_raw["loan_age_days"] >= loan_age_range[0]) & (filtered_raw["loan_age_days"] <= loan_age_range[1])
    ]

filtered_encoded = encode_dataframe(filtered_raw)

# === TABS ===
tab_explore, tab_score = st.tabs(["📊 Exploration", "🧠 Default Risk Scoring"])

with tab_explore:
    st.subheader("🔑 Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    default_rate = filtered_encoded[TARGET].mean() if TARGET in filtered_encoded.columns else 0
    c1.metric("Total Loans", f"{len(filtered_encoded):,}")
    c2.metric("Default Rate", f"{default_rate:.2%}")
    c3.metric("Avg Loan Age (days)", f"{filtered_raw['loan_age_days'].mean():.1f}" if "loan_age_days" in filtered_raw.columns else "N/A")
    c4.metric("Unique Sectors", filtered_raw["sector"].nunique() if "sector" in filtered_raw.columns else 0)

    st.markdown("## Overview & Breakdown")
    with st.container():
        left, right = st.columns([2, 1])
        with left:
            st.markdown("### Default Status")
            if TARGET in filtered_encoded.columns:
                fig = px.pie(
                    filtered_encoded,
                    names=TARGET,
                    title="Defaults vs Non-defaults",
                    hole=0.35,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Default Status Kind (raw)")
            if "Default_status_kind" in filtered_raw.columns:
                kind_df = (
                    filtered_raw["Default_status_kind"]
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
        with right:
            st.markdown("### Default Rate Over Time")
            if "report_date" in filtered_raw.columns and TARGET in filtered_encoded.columns:
                time_series = (
                    filtered_raw.assign(report_date=pd.to_datetime(filtered_raw["report_date"]).dt.date)
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

    st.markdown("## 🧩 Segment Performance")
    s1, s2 = st.columns(2)
    with s1:
        if "sector" in filtered_raw.columns:
            sector_perf = (
                filtered_encoded.groupby(filtered_encoded["sector"])
                .agg(default_rate=(TARGET, "mean"), count=(TARGET, "size"))
                .reset_index()
                .sort_values("default_rate", ascending=False)
            )
            fig_sector = px.bar(
                sector_perf,
                x="sector",
                y="default_rate",
                color="count",
                title="Default Rate by Sector (encoded)",
                hover_data={"count": True, "default_rate": ":.1%"},
            )
            fig_sector.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_sector, use_container_width=True)
    with s2:
        if "FACILITY_TYPE" in filtered_raw.columns:
            fac_perf = (
                filtered_encoded.groupby(filtered_encoded["FACILITY_TYPE"])
                .agg(default_rate=(TARGET, "mean"), count=(TARGET, "size"))
                .reset_index()
                .sort_values("default_rate", ascending=False)
            )
            fig_fac = px.bar(
                fac_perf.head(15),
                x="FACILITY_TYPE",
                y="default_rate",
                title="Default Rate by Facility Type",
                hover_data={"count": True},
            )
            fig_fac.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_fac, use_container_width=True)

with tab_score:
    st.subheader("🧠 Default Risk Scoring")
    st.caption("Raw inputs are label-encoded before scoring. Batch uses the filtered slice.")

    # === MODEL LOAD ===
    @st.cache_resource
    def load_model(path):
        return joblib.load(path)

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model '{MODEL_PATH}' not found.")
        st.stop()

    try:
        model = load_model(MODEL_PATH)
        st.success("✅ Model loaded.")
    except Exception as e:
        st.error("Failed to load model. If it embeds imblearn/SMOTE you may need Python 3.11 with matching versions.")
        st.exception(e)
        st.stop()

    threshold = st.slider("Default probability threshold", 0.0, 1.0, 0.5, 0.01)

    # === SINGLE LOAN SCORING ===
    st.markdown("### Single Loan Scoring")
    example_raw = filtered_raw.copy()
    for c in LEAK_COLS + [TARGET] + DROP_REDUNDANT:
        if c in example_raw.columns:
            example_raw = example_raw.drop(columns=[c])
    if example_raw.empty:
        st.warning("No features available for single scoring.")
    else:
        input_vals = {}
        with st.form("single_score_form"):
            for col in example_raw.columns:
                sample = example_raw[col].dropna().iloc[0] if not example_raw[col].dropna().empty else ""
                if pd.api.types.is_numeric_dtype(example_raw[col]):
                    input_vals[col] = st.number_input(col, value=float(sample) if sample != "" else 0.0, key=f"num_{col}")
                else:
                    input_vals[col] = st.text_input(col, value=str(sample), key=f"txt_{col}")
            submitted = st.form_submit_button("Score Loan")
        if submitted:
            input_df = pd.DataFrame([input_vals])
            for col in categorical_cols:
                if col in input_df.columns:
                    input_df[col] = input_df[col].astype(str).fillna("___MISSING___")
                    input_df[col] = label_encoders[col].transform(input_df[col])
            for c in LEAK_COLS + [TARGET] + DROP_REDUNDANT:
                if c in input_df.columns:
                    input_df = input_df.drop(columns=[c])
            input_df = input_df.replace("", np.nan)
            try:
                prob = model.predict_proba(input_df)[:, 1][0]
                label = int(prob >= threshold)
                st.success(f"Default probability: {prob:.3f} → Predicted label: {label}")
            except Exception as e:
                st.error("Prediction failed; check input formatting and feature alignment.")
                st.text(traceback.format_exc())

    # === BATCH SCORING ON FILTERED SLICE ===
    st.markdown("### Batch Scoring (Filtered Slice)")
    if filtered_encoded.empty:
        st.warning("Filtered slice is empty; nothing to score.")
    else:
        try:
            preds = model.predict_proba(filtered_encoded.drop(columns=[TARGET], errors="ignore"))[:, 1]
            output = filtered_raw.copy()
            output["default_probability"] = preds
            output["predicted_default"] = (preds >= threshold).astype(int)
            st.dataframe(output.head(50))
            st.download_button(
                "Download scored filtered slice",
                output.to_csv(index=False).encode("utf-8"),
                "scored_filtered_loans.csv",
                "text/csv",
            )
        except Exception as e:
            st.error("Batch scoring failed on filtered slice.")
            st.text(traceback.format_exc())

# === FOOTER ===
st.markdown(
    """
---
**Notes:**  
• Categorical inputs are label-encoded exactly as in training.  
• The model must embed the fitted preprocessing (so raw label-encoded inputs work) and preferably not depend on imblearn for Python 3.13.  
• If you retain the original pipeline with SMOTE, deploy under Python 3.11 with pinned `scikit-learn==1.6.1` and `imbalanced-learn==0.11.0`.  
"""
)
