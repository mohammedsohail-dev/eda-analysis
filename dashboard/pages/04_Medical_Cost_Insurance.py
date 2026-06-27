import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from eda_utils import (
    data_quality_report, duplicate_summary, missing_value_chart,
    descriptive_stats, outlier_report_iqr, outlier_boxplots,
    correlation_heatmap,
)

st.set_page_config(page_title="Medical Cost Insurance", layout="wide")
st.title("Medical Insurance Cost Analysis")
st.markdown("What drives insurance premiums? Age, BMI, smoking status & region.")

# ── LOAD DATA ─────────────────────────────────────────────────────────────

DATA = "../data/medical-cost-insurance.csv.gz"
df = pd.read_csv(DATA)

# ── DATA CLEANING ─────────────────────────────────────────────────────────

# Check for negative/zero charges (shouldn't exist in real data)
df = df[df['charges'] > 0].copy()

# BMI outliers — flag but don't remove; keep for analysis
# WHO standard: underweight <18.5, normal 18.5-24.9, overweight 25-29.9, obese >=30
df['bmi_category'] = pd.cut(df['bmi'],
                             bins=[0, 18.5, 25, 30, 60],
                             labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────

# Age groups
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100],
                          labels=['<25', '25-34', '35-44', '45-54', '55-64', '65+'])

# Interaction feature: smoker + bmi combo
df['smoker_bmi'] = df['smoker'] + '_' + df['bmi_category'].astype(str)

# Log-transform charges (cost data is typically right-skewed)
df['log_charges'] = np.log1p(df['charges'])

# Children flag (has at least one child)
df['has_children'] = (df['children'] > 0).astype(int)

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────

st.sidebar.header("Filters")
regions = st.sidebar.multiselect("Region", df['region'].unique(), default=None)
smoker_filter = st.sidebar.selectbox("Smoker", ["All", "yes", "no"])
if regions:
    df = df[df['region'].isin(regions)]
if smoker_filter != "All":
    df = df[df['smoker'] == smoker_filter]

# ── KPI METRICS ───────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Patients", f"{len(df):,}")
col2.metric("Avg Cost", f"${df['charges'].mean():,.0f}")
col3.metric("Median Cost", f"${df['charges'].median():,.0f}")
col4.metric("Avg Age", f"{df['age'].mean():.1f}")
col5.metric("Avg BMI", f"{df['bmi'].mean():.1f}")

# ── TABS ──────────────────────────────────────────────────────────────────

tab_names = [
    " Data Quality", " Statistics", " Outliers",
    "Cost Drivers", "Age vs Cost", "BMI vs Cost", "By Region",
]
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_names)

# ── TAB 1: DATA QUALITY ──────────────────────────────────────────────────

with tab1:
    st.subheader("Data Quality Report")

    dup = duplicate_summary(df)
    dq_cols = st.columns(3)
    dq_cols[0].metric("Total Rows", f"{dup['Total Rows']:,}")
    dq_cols[1].metric("Duplicate Rows", dup["Duplicate Rows"])
    dq_cols[2].metric("Duplicate %", f"{dup['Duplicate %']}%")

    dq = data_quality_report(df)
    st.dataframe(dq, width="stretch", hide_index=True)

    st.plotly_chart(missing_value_chart(df), width="stretch")

# ── TAB 2: STATISTICAL SUMMARY ────────────────────────────────────────────

with tab2:
    st.subheader("Descriptive Statistics")
    num_cols = ['age', 'bmi', 'children', 'charges']
    stats_df = descriptive_stats(df, num_cols)
    st.dataframe(stats_df, width="stretch")

    st.subheader("Cost Distribution by Smoker Status")
    fig = px.histogram(df, x='charges', color='smoker', barmode='overlay',
                       nbins=50, opacity=0.5, title="Insurance Cost Distribution")
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Insight**: Cost distribution is heavily right-skewed. "
                "Smokers have a separate high-cost cluster starting around $15,000.")

# ── TAB 3: OUTLIER ANALYSIS ───────────────────────────────────────────────

with tab3:
    st.subheader("Outlier Detection (IQR Method)")
    o_cols = ['age', 'bmi', 'children', 'charges']
    o_report = outlier_report_iqr(df, o_cols)
    st.dataframe(o_report, width="stretch", hide_index=True)

    st.info("**Observation**: `charges` has significant high-end outliers — "
            "these are primarily smokers with high BMI, which is expected.")

    fig = outlier_boxplots(df, o_cols)
    st.plotly_chart(fig, width="stretch")

    # Smoker-specific outliers
    st.subheader("Charges by Smoker Status")
    fig2 = px.box(df, x='smoker', y='charges', color='smoker',
                  title="Insurance Cost Outliers by Smoker")
    st.plotly_chart(fig2, width="stretch")

# ── TAB 4: COST DRIVERS (existing) ────────────────────────────────────────

with tab4:
    st.subheader("Average Cost: Smoker vs Non-Smoker")
    grp = df.groupby('smoker').agg(
        avg_charge=('charges', 'mean'),
        median_charge=('charges', 'median'),
        count=('charges', 'count')
    ).reset_index()
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(grp, x='smoker', y='avg_charge', color='smoker',
                     title="Average Cost")
        st.plotly_chart(fig, width="stretch")
    with col_b:
        fig2 = px.bar(grp, x='smoker', y='median_charge', color='smoker',
                      title="Median Cost")
        st.plotly_chart(fig2, width="stretch")

    # Cost by smoker + children
    grp2 = df.groupby(['smoker', 'has_children']).agg(cost=('charges', 'mean')).reset_index()
    fig3 = px.bar(grp2, x='smoker', y='cost', color='has_children', barmode='group',
                  title="Cost by Smoker & Children", labels={'cost': 'Avg Cost'})
    st.plotly_chart(fig3, width="stretch")

    st.markdown("**Insight**: Smokers pay ~4x more than non-smokers on average. "
                "Having children also increases cost, but the smoking effect dominates.")

# ── TAB 5: AGE VS COST (existing) ─────────────────────────────────────────

with tab5:
    fig = px.scatter(df, x='age', y='charges', color='smoker', size='bmi',
                     title="Age vs Insurance Cost",
                     labels={'charges': 'Cost'}, opacity=0.6)
    st.plotly_chart(fig, width="stretch")

    # Age group aggregation
    grp = df.groupby(['age_group', 'smoker'], observed=False).agg(
        cost=('charges', 'mean')
    ).reset_index()
    fig2 = px.line(grp, x='age_group', y='cost', color='smoker', markers=True,
                   title="Average Cost by Age Group",
                   labels={'cost': 'Avg Cost', 'age_group': ''})
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: Cost increases linearly with age for non-smokers. "
                "For smokers, the increase is steeper and more variable.")

# ── TAB 6: BMI VS COST (existing) ─────────────────────────────────────────

with tab6:
    fig = px.scatter(df, x='bmi', y='charges', color='smoker', facet_col='sex',
                     title="BMI vs Cost by Smoker & Sex", opacity=0.5)
    st.plotly_chart(fig, width="stretch")

    # BMI category breakdown
    grp = df.groupby(['bmi_category', 'smoker'], observed=False).agg(
        cost=('charges', 'mean'),
        count=('charges', 'count')
    ).reset_index()
    fig2 = px.bar(grp, x='bmi_category', y='cost', color='smoker', barmode='group',
                  title="Average Cost by BMI Category & Smoker",
                  labels={'cost': 'Avg Cost'})
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: BMI alone has mild impact on cost for non-smokers. "
                "But obese smokers pay the highest premiums — the interaction is key.")

# ── TAB 7: BY REGION (existing) ───────────────────────────────────────────

with tab7:
    grp = df.groupby(['region', 'smoker']).agg(
        avg_charge=('charges', 'mean'),
        count=('charges', 'count')
    ).reset_index()
    fig = px.bar(grp, x='region', y='avg_charge', color='smoker', barmode='group',
                 title="Average Cost by Region & Smoker",
                 labels={'avg_charge': 'Avg Cost'})
    st.plotly_chart(fig, width="stretch")

    # Correlation heatmap
    corr_cols = ['age', 'bmi', 'children', 'charges']
    st.plotly_chart(correlation_heatmap(df, corr_cols), width="stretch")

    st.markdown("**Insight**: Southeast has the highest average costs, driven by "
                "a higher proportion of smokers and higher BMI in that region.")

# ── RAW DATA ──────────────────────────────────────────────────────────────

with st.expander("View Raw Data"):
    st.dataframe(df)
