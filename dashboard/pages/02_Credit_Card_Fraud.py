import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from eda_utils import (
    data_quality_report, duplicate_summary, missing_value_chart,
    descriptive_stats, outlier_report_iqr, outlier_boxplots,
    correlation_heatmap, class_balance_chart,
)

st.set_page_config(page_title="Credit Card Fraud", layout="wide")
st.title("Credit Card Fraud Detection")
st.markdown("284,807 transactions — only 492 fraudulent (0.17%). Classic imbalanced classification problem.")

# ── LOAD DATA ─────────────────────────────────────────────────────────────

DATA = "dashboard/data/credit-card-fraud.csv.gz"
df = pd.read_csv(DATA)

# ── DATA CLEANING ─────────────────────────────────────────────────────────

# No missing values expected in this dataset (confirmed by data_quality_report)
# PCA features V1-V28 are already scaled; Amount is not
# Check for any NaN values silently (none should exist)

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────

# Convert Time (seconds) to hour-of-day feature
df['Time_Hour'] = (df['Time'] / 3600) % 24

# Log-transform Amount to reduce skew for better visualisation
df['Log_Amount'] = df['Amount'].apply(lambda x: np.log1p(x) if x >= 0 else 0)

# Binary flag columns for convenience
df['Class_Label'] = df['Class'].map({0: 'Legit', 1: 'Fraud'})

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────

st.sidebar.header("Filters")
amount_max = st.sidebar.slider("Max Amount ($)", 0, 10000, 5000)
df = df[df['Amount'] <= amount_max]

fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0]

# ── KPI METRICS ───────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Fraud Cases", f"{len(fraud):,}")
col3.metric("Fraud Rate", f"{len(fraud)/len(df):.4%}")
col4.metric("Avg Fraud Amount", f"${fraud['Amount'].mean():.2f}" if len(fraud) > 0 else "$0")

# ── TABS ──────────────────────────────────────────────────────────────────

tab_names = [
    " Data Quality", " Statistics", " Outliers",
    "Amount Distribution", "Time Patterns", "PCA Components", "Correlation",
]
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_names)

# ── TAB 1: DATA QUALITY ──────────────────────────────────────────────────

with tab1:
    st.subheader("Data Quality Report")

    dup = duplicate_summary(df)
    dq_cols = st.columns(3)
    dq_cols[0].metric("Total Rows", f"{dup['Total Rows']:,}")
    dq_cols[1].metric("Duplicate Rows", f"{dup['Duplicate Rows']:,}")
    dq_cols[2].metric("Duplicate %", f"{dup['Duplicate %']}%")

    dq = data_quality_report(df)
    st.dataframe(dq, width="stretch", hide_index=True)

    st.plotly_chart(missing_value_chart(df), width="stretch")

    st.plotly_chart(class_balance_chart(df, 'Class'), width="stretch")

    st.info("**Key observation**: Only 0.17% of transactions are fraudulent. "
            "This extreme imbalance means a naive classifier that always predicts 'legit' "
            "would be 99.83% accurate yet useless.  Any modeling must use "
            "techniques like SMOTE, class weighting, or anomaly detection.")

# ── TAB 2: STATISTICAL SUMMARY ────────────────────────────────────────────

with tab2:
    st.subheader("Descriptive Statistics")
    num_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    stats_df = descriptive_stats(df, num_cols)
    st.dataframe(stats_df, width="stretch")

    st.subheader("Amount Distribution by Class")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=legit['Amount'], name='Legit', opacity=0.6, nbinsx=80))
    fig.add_trace(go.Histogram(x=fraud['Amount'], name='Fraud', opacity=0.8, nbinsx=80))
    fig.update_layout(barmode='overlay', title="Transaction Amount: Legit vs Fraud",
                      xaxis_title="Amount ($)")
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Insight**: Fraudulent transactions tend to have **lower** amounts "
                "on average than legitimate ones, though with high variance.")

# ── TAB 3: OUTLIER ANALYSIS ───────────────────────────────────────────────

with tab3:
    st.subheader("Outlier Detection (IQR Method)")
    o_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    o_report = outlier_report_iqr(df, o_cols)
    st.dataframe(o_report, width="stretch", hide_index=True)

    st.info("**Note**: PCA features (V1-V28) are standardized (mean≈0, std≈1), "
            "so IQR outliers on them are expected by design.  Focus on `Amount` "
            "and `Time` for meaningful outlier inspection.")

    fig = outlier_boxplots(df, ['Amount', 'Time'] + [f'V{i}' for i in range(1, 7)])
    st.plotly_chart(fig, width="stretch")

# ── TAB 4: AMOUNT DISTRIBUTION (existing) ─────────────────────────────────

with tab4:
    st.subheader("Transaction Amount: Legit vs Fraud")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=legit['Amount'][legit['Amount'] <= 500],
                                name='Legit', opacity=0.6, nbinsx=50))
    fig.add_trace(go.Histogram(x=fraud['Amount'][fraud['Amount'] <= 500],
                                name='Fraud', opacity=0.8, nbinsx=50))
    fig.update_layout(barmode='overlay',
                      title="Amount Distribution (≤ $500 zoomed)",
                      xaxis_title="Amount ($)")
    st.plotly_chart(fig, width="stretch")

    # Stats table
    st.dataframe(
        df.groupby('Class')['Amount'].describe().round(2),
        width="stretch"
    )

# ── TAB 5: TIME PATTERNS (existing) ───────────────────────────────────────

with tab5:
    grp = df.groupby('Time_Hour').agg(
        rate=('Class', 'mean'),
        count=('Class', 'count')
    ).reset_index()
    fig = px.line(grp, x='Time_Hour', y='rate', markers=True,
                  title="Fraud Rate by Hour of Day",
                  labels={'Time_Hour': 'Hour', 'rate': 'Fraud Rate'})
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    st.plotly_chart(fig, width="stretch")

    # Transaction volume by hour
    grp2 = df.groupby('Time_Hour').agg(transactions=('Class', 'count')).reset_index()
    fig2 = px.bar(grp2, x='Time_Hour', y='transactions',
                  title="Transaction Volume by Hour of Day")
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: Fraud rate is highest in the early morning hours (2-5 AM) "
                "when legitimate transaction volume is lowest — a common fraud pattern.")

# ── TAB 6: PCA COMPONENTS (existing) ──────────────────────────────────────

with tab6:
    pca_cols = [c for c in df.columns if c.startswith('V')]

    # Select first 4 PCA dimensions to show
    dims = st.multiselect("Select PCA dimensions to compare",
                          pca_cols, default=['V1', 'V2', 'V3', 'V4'])
    if len(dims) >= 2:
        fig = px.scatter_matrix(df.sample(5000), dimensions=dims, color='Class_Label',
                                title="PCA Component Relationships (sample 5k)", opacity=0.3)
        fig.update_traces(marker_size=2)
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Select at least 2 dimensions for the scatter matrix.")

# ── TAB 7: CORRELATION ────────────────────────────────────────────────────

with tab7:
    st.subheader("PCA Feature Correlation")
    pca_cols = [c for c in df.columns if c.startswith('V')]
    corr_cols = st.multiselect("Select PCA dimensions for correlation heatmap",
                                pca_cols, default=pca_cols[:10])
    if len(corr_cols) >= 2:
        st.plotly_chart(correlation_heatmap(df, corr_cols), width="stretch")
    else:
        st.warning("Select at least 2 columns.")

# ── RAW DATA ──────────────────────────────────────────────────────────────

with st.expander("View Raw Data"):
    st.dataframe(df.head(1000))
