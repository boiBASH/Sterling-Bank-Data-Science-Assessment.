import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# === CONFIG ===
TARGET = "Default_status"
DROP_REDUNDANT = ["Default_status_kind", "ARR_STATUS", "LINE_DESC"]
LEAK_COLS = ["DAYS_TO_MATURITY", "CONTRACT_MAT_DATE", "report_date", "PayinAccount_Last_LOD_Date"]
MODEL_PATH = "light_rf_model.pkl"
DATA_PATH = "cleaned_loan_data.xlsx"
LOGO_PATH = "sterling bank logo.png"
DEFAULT_THRESHOLD = 0.5

st.set_page_config(page_title="Loan Explorer & Default Scoring", layout="wide")

# === HEADER ===
col_logo, col_title = st.columns([1, 7])
with col_logo:
    try:
        st.image(LOGO_PATH, width=80)
    except Exception:
        st.markdown("**Sterling Bank**")
with col_title:
    st.markdown("<h1 style='margin:0;'>📊 Loan Explorer & Default Risk Scoring</h1>", unsafe_allow_html=True)
    st.markdown("Explore loans and get a customer-facing default status.", unsafe_allow_html=True)

# === HELPERS ===
def make_cols_unique(df: pd.DataFrame) -> pd.DataFrame:
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

def sanitize_df_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()]

# === LOAD DATA ===
@st.cache_data
def load_raw_data(path):
    return pd.read_excel(path)

if not os.path.exists(DATA_PATH):
    st.error(f"Missing data file '{DATA_PATH}'. Place {DATA_PATH} in repo root.")
    st.stop()
df_raw = load_raw_data(DATA_PATH)
df_raw = make_cols_unique(df_raw)

# === BUILD LABEL ENCODERS ON THE FLY ===
@st.cache_resource
def build_label_encoders(df: pd.DataFrame):
    drop_for_encoding = DROP_REDUNDANT + [TARGET]
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    for rm in drop_for_encoding:
        if rm in categorical_cols:
            categorical_cols.remove(rm)
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str).fillna("___MISSING___"))
        encoders[col] = le
    return encoders

label_encoders = build_label_encoders(df_raw)
categorical_enc_cols = list(label_encoders.keys())

def encode_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    for col in categorical_enc_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("___MISSING___")
            try:
                df[col] = label_encoders[col].transform(df[col])
            except Exception:
                df[col] = -1
    for c in DROP_REDUNDANT:
        if c in df.columns:
            df = df.drop(columns=c)
    return df

# Encoded for exploration / batch scoring
df_encoded = encode_df(df_raw)
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
filtered_raw = make_cols_unique(filtered_raw)
filtered_encoded = encode_df(filtered_raw)
filtered_encoded = make_cols_unique(filtered_encoded)

# === MODEL LOADING ===
@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found.")
    st.stop()

try:
    model = load_model(MODEL_PATH)
    st.success("✅ Model loaded.")
except Exception as e:
    st.error("Failed to load model. If it relies on imblearn/SMOTE, deploy under Python 3.11 with required versions.")
    st.exception(e)
    st.stop()

# === TABS ===
tab_explore, tab_score = st.tabs(["📊 Exploration", "🧠 Default Risk Scoring"])

with tab_explore:
    st.subheader("🔑 Key Metrics")
    a, b, c, d = st.columns(4)
    default_rate = filtered_encoded[TARGET].mean() if TARGET in filtered_encoded.columns else 0
    a.metric("Total Loans", f"{len(filtered_encoded):,}")
    b.metric("Default Rate", f"{default_rate:.2%}")
    c.metric("Avg Loan Age (days)", f"{filtered_raw['loan_age_days'].mean():.1f}" if "loan_age_days" in filtered_raw.columns else "N/A")
    d.metric("Unique Sectors", filtered_raw["sector"].nunique() if "sector" in filtered_raw.columns else 0)

    st.markdown("## Overview & Breakdown")
    with st.container():
        left, right = st.columns([2, 1])
        with left:
            st.markdown("### Default Status Distribution")
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
                kind_series = filtered_raw["Default_status_kind"].dropna()
                kind_df = kind_series.value_counts().reset_index()
                kind_df.columns = ["kind", "count"]
                kind_df = sanitize_df_for_plot(kind_df)
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
                fig_time = px.line(time_series, x="report_date", y="default_rate", title="Default Rate Over Time", markers=True)
                fig_time.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("## Segment Performance")
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
                title="Default Rate by Sector",
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
    st.markdown("Interactive single-loan input. Numeric via sliders; categorical via selectboxes. Customer-facing status below.")

    threshold = st.slider("Default probability threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)

    # === SINGLE LOAN SCORING ===
    st.markdown("### Single Loan Scoring")
    input_cols = [c for c in df_raw.columns if c not in LEAK_COLS + [TARGET] + DROP_REDUNDANT]
    numeric_inputs = [c for c in input_cols if pd.api.types.is_numeric_dtype(df_raw[c])]
    categorical_inputs = [c for c in input_cols if not pd.api.types.is_numeric_dtype(df_raw[c])]

    sample_row = filtered_raw.iloc[0] if not filtered_raw.empty else df_raw.iloc[0]

    with st.form("single_score_form"):
        st.markdown("#### Numeric Inputs")
        num_vals = {}
        for col in numeric_inputs:
            series = df_raw[col].dropna()
            if series.empty:
                continue
            vmin = float(series.quantile(0.01))
            vmax = float(series.quantile(0.99))
            default = float(sample_row[col]) if pd.notna(sample_row[col]) else (vmin + vmax) / 2
            if vmax > vmin:
                num_vals[col] = st.slider(col, min_value=vmin, max_value=vmax, value=default, step=(vmax - vmin) / 100)
            else:
                num_vals[col] = st.number_input(col, value=default)
        st.markdown("#### Categorical Inputs")
        cat_vals = {}
        for col in categorical_inputs:
            if col not in df_raw.columns:
                continue
            opts = sorted(pd.Series(df_raw[col].dropna().astype(str).unique()).tolist())
            if not opts:
                continue
            default = str(sample_row[col]) if pd.notna(sample_row[col]) else opts[0]
            index = opts.index(default) if default in opts else 0
            cat_vals[col] = st.selectbox(col, options=opts, index=index)
        submitted = st.form_submit_button("Score Loan")

    if submitted:
        raw_input = {}
        raw_input.update(num_vals)
        raw_input.update(cat_vals)
        input_df = pd.DataFrame([raw_input])
        input_df = encode_df(input_df)
        for c in LEAK_COLS + [TARGET] + DROP_REDUNDANT:
            if c in input_df.columns:
                input_df = input_df.drop(columns=[c])
        input_df = input_df.replace("", np.nan)
        try:
            prob = model.predict_proba(input_df)[:, 1][0]
            label = int(prob >= threshold)
            status = "Default" if label == 1 else "No Default"
            banner = "⚠️ High risk of default" if label == 1 else "✅ Low risk of default"
            st.metric("Predicted Status", status)
            st.success(f"Default probability: {prob:.3f}")
            st.info(banner)
        except Exception as e:
            st.error("Prediction failed; ensure inputs align with training schema.")
            st.text(traceback.format_exc())

    # === BATCH SCORING ===
    st.markdown("### Batch Scoring (Filtered Slice)")
    if filtered_encoded.empty:
        st.warning("Filtered slice is empty.")
    else:
        try:
            X_batch = filtered_encoded.drop(columns=[TARGET], errors="ignore")
            probs = model.predict_proba(X_batch)[:, 1]
            output = filtered_raw.copy()
            output["default_probability"] = probs
            output["predicted_default"] = (probs >= threshold).astype(int)
            output["status"] = output["predicted_default"].map({1: "Default", 0: "No Default"})
            st.dataframe(output.head(50))
            st.download_button(
                "Download scored filtered slice",
                output.to_csv(index=False).encode("utf-8"),
                "scored_filtered_loans.csv",
                "text/csv",
            )
        except Exception as e:
            st.error("Batch scoring failed.")
            st.text(traceback.format_exc())

# === FOOTER ===
st.markdown(
    """
---
**Customer-facing default status logic:**  
• Probability >= threshold → **Default** (high risk) with a warning.  
• Probability < threshold → **No Default** (low risk).  

**Notes:**  
• Categorical features are label-encoded on the fly to match training.  
• The model expects preprocessed numeric inputs (imputer + scaler embedded).  
• If you kept a pipeline with SMOTE/imbalanced-learn, run under Python 3.11 with pinned `scikit-learn==1.6.1` and `imbalanced-learn==0.11.0`.  
"""
)
