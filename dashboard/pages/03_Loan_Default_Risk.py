import streamlit as st
import pandas as pd
import plotly.express as px
from eda_utils import (
    data_quality_report, duplicate_summary, missing_value_chart,
    descriptive_stats, outlier_report_iqr, outlier_boxplots,
    correlation_heatmap, class_balance_chart,
)

st.set_page_config(page_title="Loan Default Risk", layout="wide")
st.title("Loan Default Risk Analysis")
st.markdown("Credit risk assessment — predict loan defaults from borrower attributes.")

# ── LOAD DATA ─────────────────────────────────────────────────────────────

DATA = "../data/loan-default.csv.gz"
df = pd.read_csv(DATA)

# ── DATA CLEANING ─────────────────────────────────────────────────────────

# emp_length values are short codes like '4', '10+', '<1' — map to numeric
emp_map = {
    '<1': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
    '6': 6, '7': 7, '8': 8, '9': 9, '10+': 10,
}
df['emp_years'] = df['emp_length'].map(emp_map)

# Drop rows where emp_length couldn't be mapped (should be none)
df = df.dropna(subset=['emp_years']).copy()

# Ensure dti is finite
df = df[df['dti'].notna()].copy()

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────

# Loan-to-income ratio — a key creditworthiness metric
df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)

# Grade as numeric for correlation analysis
grade_order = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df['grade_num'] = df['grade'].map(grade_order)

# Interest rate brackets
df['int_rate_bin'] = pd.cut(df['int_rate'], bins=[0, 8, 12, 16, 20, 30],
                             labels=['<8%', '8-12%', '12-16%', '16-20%', '20%+'])

# Home ownership binary flag
df['has_mortgage'] = (df['home_ownership'] == 'MORTGAGE').astype(int)

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────

st.sidebar.header("Filters")
grades = st.sidebar.multiselect("Grade", sorted(df['grade'].unique()), default=None)
purposes = st.sidebar.multiselect("Purpose", df['purpose'].unique(), default=None)
if grades:
    df = df[df['grade'].isin(grades)]
if purposes:
    df = df[df['purpose'].isin(purposes)]

# ── KPI METRICS ───────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Loans", f"{len(df):,}")
col2.metric("Default Rate", f"{df['default'].mean():.1%}")
col3.metric("Avg Loan Amount", f"${df['loan_amnt'].mean():,.0f}")
col4.metric("Avg Interest Rate", f"{df['int_rate'].mean():.1f}%")
col5.metric("Avg DTI", f"{df['dti'].mean():.1f}%")

# ── TABS ──────────────────────────────────────────────────────────────────

tab_names = [
    " Data Quality", " Statistics", " Outliers",
    "Default by Grade", "Income vs Loan", "Interest by Purpose", "DTI Analysis",
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

    st.plotly_chart(class_balance_chart(df, 'default'), width="stretch")

# ── TAB 2: STATISTICAL SUMMARY ────────────────────────────────────────────

with tab2:
    st.subheader("Descriptive Statistics")
    num_cols = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
                'emp_years', 'loan_to_income']
    stats_df = descriptive_stats(df, num_cols)
    st.dataframe(stats_df, width="stretch")

    st.subheader("Distributions")
    selected = st.multiselect("Select columns", num_cols, default=['loan_amnt', 'int_rate', 'dti'])
    if selected:
        fig = px.histogram(df.melt(value_vars=selected, var_name="Variable", value_name="Value"),
                           x="Value", facet_row="Variable", nbins=40)
        fig.update_layout(height=250 * len(selected))
        st.plotly_chart(fig, width="stretch")

# ── TAB 3: OUTLIER ANALYSIS ───────────────────────────────────────────────

with tab3:
    st.subheader("Outlier Detection (IQR Method)")
    o_cols = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'emp_years']
    o_report = outlier_report_iqr(df, o_cols)
    st.dataframe(o_report, width="stretch", hide_index=True)

    st.info("**Observation**: `annual_inc` and `loan_amnt` have many high-end outliers — "
            "high-income borrowers taking very large loans are rare but worth separate analysis.")

    fig = outlier_boxplots(df, o_cols)
    st.plotly_chart(fig, width="stretch")

# ── TAB 4: DEFAULT BY GRADE (existing) ────────────────────────────────────

with tab4:
    grp = df.groupby('grade').agg(
        default_rate=('default', 'mean'),
        count=('default', 'count')
    ).reset_index()
    grp['rate_label'] = grp['default_rate'].apply(lambda x: f"{x:.1%}")
    fig = px.bar(grp, x='grade', y='default_rate', color='count',
                 color_continuous_scale='RdYlGn_r',
                 title="Default Rate by Loan Grade (A = safest, G = riskiest)",
                 labels={'default_rate': 'Default Rate'}, text='rate_label')
    st.plotly_chart(fig, width="stretch")

    # Default by term
    grp2 = df.groupby('term').agg(rate=('default', 'mean')).reset_index()
    fig2 = px.bar(grp2, x='term', y='rate', color='rate',
                  color_continuous_scale='RdYlGn_r',
                  title="Default Rate by Loan Term",
                  labels={'rate': 'Default Rate'})
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: Default rate increases almost monotonically from grade A to G. "
                "60-month loans default at roughly double the rate of 36-month loans.")

# ── TAB 5: INCOME VS LOAN (existing) ──────────────────────────────────────

with tab5:
    fig = px.scatter(df, x='annual_inc', y='loan_amnt', color='default', opacity=0.4,
                     title="Annual Income vs Loan Amount",
                     labels={'annual_inc': 'Annual Income', 'loan_amnt': 'Loan Amount'})
    st.plotly_chart(fig, width="stretch")

    # Loan-to-income by default status
    fig2 = px.box(df, x='default', y='loan_to_income',
                  title="Loan-to-Income Ratio by Default Status",
                  labels={'default': 'Default', 'loan_to_income': 'Loan / Income'})
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: Defaulters tend to have higher loan-to-income ratios on average, "
                "confirming that over-leverage is a key default driver.")

# ── TAB 6: INTEREST BY PURPOSE (existing) ─────────────────────────────────

with tab6:
    grp = df.groupby('purpose').agg(
        avg_rate=('int_rate', 'mean'),
        default_rate=('default', 'mean')
    ).reset_index()
    grp = grp.sort_values('avg_rate', ascending=True)
    fig = px.bar(grp, x='purpose', y='avg_rate', color='default_rate',
                 color_continuous_scale='RdYlGn_r',
                 title="Avg Interest Rate by Loan Purpose",
                 labels={'avg_rate': 'Avg Rate', 'purpose': ''})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Insight**: Small business and debt consolidation loans carry the highest "
                "interest rates and default rates.  Credit cards and car loans are safer.")

# ── TAB 7: DTI ANALYSIS (existing) ────────────────────────────────────────

with tab7:
    fig = px.histogram(df, x='dti', color=df['default'].map({1: 'Default', 0: 'Paid'}),
                       barmode='overlay', nbins=40, opacity=0.5,
                       title="Debt-to-Income Ratio Distribution")
    st.plotly_chart(fig, width="stretch")

    # DTI bins vs default rate
    df['dti_bin'] = pd.cut(df['dti'], bins=range(0, 45, 5))
    grp = df.groupby('dti_bin', observed=False).agg(rate=('default', 'mean')).reset_index()
    grp['dti_bin'] = grp['dti_bin'].astype(str)
    fig2 = px.bar(grp, x='dti_bin', y='rate',
                  title="Default Rate by DTI Bucket",
                  labels={'dti_bin': 'DTI Range', 'rate': 'Default Rate'})
    st.plotly_chart(fig2, width="stretch")

# ── RAW DATA ──────────────────────────────────────────────────────────────

with st.expander("View Raw Data"):
    st.dataframe(df.head(1000))
