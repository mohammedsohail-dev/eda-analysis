import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from eda_utils import (
    data_quality_report, duplicate_summary, missing_value_chart,
    descriptive_stats, outlier_report_iqr, outlier_boxplots,
    correlation_heatmap, class_balance_chart,
)

st.set_page_config(page_title="Bank Marketing", layout="wide")
st.title("Bank Marketing Campaign Analysis")
st.markdown("Predict term deposit subscriptions from direct marketing campaigns.")

# ── LOAD DATA ─────────────────────────────────────────────────────────────

DATA = "../data/bank-marketing.csv.gz"
df = pd.read_csv(DATA)

# ── DATA CLEANING ─────────────────────────────────────────────────────────

# Create binary target for easier analysis
df['y_binary'] = (df['y'] == 'yes').astype(int)

# Check for any remaining missing values in key columns
# poutcome and contact have natural missingness (not contacted yet)
# pdays = 999 means "not previously contacted" — keep as-is

# Remove rows where duration is 0 (no contact happened — these are non-events)
df = df[df['duration'] > 0].copy()

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────

# Age groups for demographic segmentation
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100],
                         labels=['<25', '25-34', '35-44', '45-54', '55-64', '65+'])

# Encode month as numeric for trend analysis
month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
             'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
df['month_num'] = df['month'].map(month_map)

# Previous contact ratio (campaign efficiency proxy)
df['prev_contact_ratio'] = df['previous'] / (df['campaign'] + 1)

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────

st.sidebar.header("Filters")
jobs = st.sidebar.multiselect("Job", sorted(df['job'].unique()), default=None)
months = st.sidebar.multiselect("Month", sorted(df['month'].unique()), default=None)
if jobs:
    df = df[df['job'].isin(jobs)]
if months:
    df = df[df['month'].isin(months)]

# ── KPI METRICS ───────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Contacts", f"{len(df):,}")
col2.metric("Subscriptions", f"{df['y_binary'].sum():,}")
col3.metric("Conversion Rate", f"{df['y_binary'].mean():.1%}")
col4.metric("Avg Call Duration", f"{df['duration'].mean():.0f}s")
col5.metric("Avg Campaign Calls", f"{df['campaign'].mean():.1f}")

# ── TABS ──────────────────────────────────────────────────────────────────

tab_names = [
    " Data Quality", " Statistics", " Outliers",
    "Conversion by Job", "Age Distribution", "Month Trends", "Economic Factors",
]
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_names)

# ── TAB 1: DATA QUALITY ──────────────────────────────────────────────────

with tab1:
    st.subheader("Data Quality Report")

    # Summary metrics
    dup = duplicate_summary(df)
    dq_cols = st.columns(3)
    dq_cols[0].metric("Total Rows", dup["Total Rows"])
    dq_cols[1].metric("Duplicate Rows", dup["Duplicate Rows"])
    dq_cols[2].metric("Duplicate %", f"{dup['Duplicate %']}%")

    # Column-level quality table
    dq = data_quality_report(df)
    st.dataframe(dq, width="stretch", hide_index=True)

    # Missing values chart
    st.plotly_chart(missing_value_chart(df), width="stretch")

    # Class balance
    st.plotly_chart(class_balance_chart(df, 'y'), width="stretch")

# ── TAB 2: STATISTICAL SUMMARY ────────────────────────────────────────────

with tab2:
    st.subheader("Descriptive Statistics")
    num_cols = ['age', 'duration', 'campaign', 'pdays', 'previous',
                'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    stats_df = descriptive_stats(df, num_cols)
    st.dataframe(stats_df, width="stretch")

    st.subheader("Distribution Histograms")
    selected = st.multiselect("Select columns to plot", num_cols, default=['age', 'duration', 'campaign', 'euribor3m'])
    if selected:
        fig = px.histogram(df.melt(value_vars=selected, var_name="Variable", value_name="Value"),
                           x="Value", facet_row="Variable", nbins=40,
                           title="Distributions")
        fig.update_layout(height=250 * len(selected))
        st.plotly_chart(fig, width="stretch")

# ── TAB 3: OUTLIER ANALYSIS ───────────────────────────────────────────────

with tab3:
    st.subheader("Outlier Detection (IQR Method)")
    outlier_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'euribor3m']
    o_report = outlier_report_iqr(df, outlier_cols)
    st.dataframe(o_report, width="stretch", hide_index=True)

    st.info("**Key observation**: `duration` and `campaign` have many high outliers. "
            "Very long calls (duration > 3000s) and very high campaign counts (> 30) "
            "are rare events worth inspecting separately.")

    fig = outlier_boxplots(df, outlier_cols)
    st.plotly_chart(fig, width="stretch")

# ── TAB 4: CONVERSION BY JOB (existing analysis) ──────────────────────────

with tab4:
    grp = df.groupby('job').agg(
        rate=('y_binary', 'mean'),
        count=('y_binary', 'count')
    ).reset_index()
    grp['rate_label'] = grp['rate'].apply(lambda x: f"{x:.1%}")
    fig = px.bar(grp, x='job', y='rate', color='count', color_continuous_scale='Blues',
                 title="Conversion Rate by Job (top = most likely to subscribe)",
                 labels={'rate': 'Rate', 'job': ''}, text='rate_label')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Insight**: Students and retired people have the highest conversion rates. "
                "Blue-collar workers are hardest to convert.")

# ── TAB 5: AGE DISTRIBUTION (existing analysis) ───────────────────────────

with tab5:
    fig = px.histogram(df, x='age', color='y', barmode='overlay', nbins=40,
                       title="Age Distribution by Subscription", opacity=0.6)
    st.plotly_chart(fig, width="stretch")

    # Age-group breakdown
    grp = df.groupby('age_group', observed=False).agg(
        rate=('y_binary', 'mean'), count=('y_binary', 'count')
    ).reset_index()
    fig2 = px.bar(grp, x='age_group', y='rate', color='count',
                  color_continuous_scale='Blues',
                  title="Conversion Rate by Age Group")
    st.plotly_chart(fig2, width="stretch")

# ── TAB 6: MONTH TRENDS (existing analysis) ───────────────────────────────

with tab6:
    month_order = ['mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    grp = df.groupby('month').agg(
        rate=('y_binary', 'mean'),
        contacts=('y_binary', 'count')
    ).reset_index()
    grp['month'] = pd.Categorical(grp['month'], categories=month_order, ordered=True)
    grp = grp.sort_values('month')
    fig = px.line(grp, x='month', y='rate', markers=True,
                  title="Conversion Rate by Month",
                  labels={'rate': 'Conversion Rate', 'month': ''})
    fig.update_traces(line=dict(width=3), marker=dict(size=10))
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Insight**: Conversion peaks in March and drops sharply by May. "
                "Campaign timing clearly matters — avoiding late spring improves results.")

# ── TAB 7: ECONOMIC FACTORS (existing analysis) ───────────────────────────

with tab7:
    fig = px.scatter(df, x='euribor3m', y='cons.price.idx', color='y', opacity=0.3,
                     title="Euribor vs Consumer Price Index (colored by subscription)",
                     labels={'euribor3m': 'Euribor 3m', 'cons.price.idx': 'CPI'})
    st.plotly_chart(fig, width="stretch")

    # Correlation with economic indicators
    econ_cols = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y_binary']
    st.plotly_chart(correlation_heatmap(df, econ_cols), width="stretch")

    st.markdown("**Insight**: Subscriptions are strongly correlated with lower Euribor rates "
                "and higher employment — economic booms boost campaign success.")

# ── RAW DATA ──────────────────────────────────────────────────────────────

with st.expander("View Raw Data"):
    st.dataframe(df.head(1000))
