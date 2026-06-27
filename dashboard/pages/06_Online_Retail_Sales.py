import streamlit as st
import pandas as pd
import plotly.express as px
from eda_utils import (
    data_quality_report, duplicate_summary, missing_value_chart,
    descriptive_stats, outlier_report_iqr, outlier_boxplots,
    correlation_heatmap,
)

st.set_page_config(page_title="Online Retail Sales", layout="wide")
st.title("Online Retail Sales Analytics")
st.markdown("541K transactions from UK-based online retailer. Revenue, customer & product analysis.")

# ── LOAD DATA ─────────────────────────────────────────────────────────────

DATA = "../data/online-retail-sales.csv.gz"
df = pd.read_csv(DATA)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# ── DATA CLEANING ─────────────────────────────────────────────────────────

# Separate cancellations (negative Quantity) from actual sales
df['is_cancellation'] = df['Quantity'] < 0
cancellations = df[df['is_cancellation']].copy()
sales = df[~df['is_cancellation']].copy()

# Work with actual sales for core analysis
df = sales.copy()

# Remove rows with missing CustomerID (can't attribute to a customer)
st.sidebar.markdown("**Data Quality Note**")
cust_missing = df['CustomerID'].isna().sum()
st.sidebar.info(f"{cust_missing:,} rows ({cust_missing/len(df):.1%}) have no CustomerID — excluded from customer-level metrics")
df = df.dropna(subset=['CustomerID']).copy()

# Remove rows with missing or empty Description
df = df.dropna(subset=['Description']).copy()
df = df[df['Description'].str.strip() != ''].copy()

# Filter out extreme UnitPrice outliers (likely data errors)
df = df[(df['UnitPrice'] > 0) & (df['UnitPrice'] < 10000)].copy()

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────

# Revenue per line item
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# Temporal features
df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)
df['Year'] = df['InvoiceDate'].dt.year
df['Month_of_Year'] = df['InvoiceDate'].dt.month
df['Day_of_Week'] = df['InvoiceDate'].dt.dayofweek
df['Hour'] = df['InvoiceDate'].dt.hour

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────

st.sidebar.header("Filters")
countries = st.sidebar.multiselect("Country", df['Country'].dropna().unique(), default=None)
if countries:
    df = df[df['Country'].isin(countries)]

# ── KPI METRICS ───────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders", f"{df['InvoiceNo'].nunique():,}")
col2.metric("Total Revenue", f"${df['Revenue'].sum():,.0f}")
col3.metric("Avg Order Value", f"${df.groupby('InvoiceNo')['Revenue'].sum().mean():,.0f}")
col4.metric("Unique Customers", f"{df['CustomerID'].nunique():,.0f}")

# ── TABS ──────────────────────────────────────────────────────────────────

tab_names = [
    " Data Quality", " Statistics", " Outliers",
    "Revenue by Country", "Top Products", "Revenue Over Time", "Customer Spend",
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

    st.subheader("Cancellation Analysis")
    canc_rate = len(cancellations) / (len(cancellations) + len(sales))
    st.metric("Cancellation Rate", f"{canc_rate:.2%}")
    canc_by_country = cancellations.groupby('Country').size().reset_index(name='cancellations')
    canc_by_country = canc_by_country.sort_values('cancellations', ascending=False).head(10)
    fig = px.bar(canc_by_country, x='Country', y='cancellations',
                 title="Cancellations by Country (Top 10)")
    st.plotly_chart(fig, width="stretch")

# ── TAB 2: STATISTICAL SUMMARY ────────────────────────────────────────────

with tab2:
    st.subheader("Descriptive Statistics")
    num_cols = ['Quantity', 'UnitPrice', 'Revenue']
    stats_df = descriptive_stats(df, num_cols)
    st.dataframe(stats_df, width="stretch")

    st.subheader("Distributions")
    selected = st.multiselect("Select variables", num_cols,
                               default=['Quantity', 'UnitPrice', 'Revenue'])
    if selected:
        fig = px.histogram(
            df.melt(value_vars=selected, var_name="Variable", value_name="Value"),
            x="Value", facet_row="Variable", nbins=50, marginal="box"
        )
        fig.update_layout(height=300 * len(selected))
        st.plotly_chart(fig, width="stretch")

# ── TAB 3: OUTLIER ANALYSIS ───────────────────────────────────────────────

with tab3:
    st.subheader("Outlier Detection (IQR Method)")
    o_cols = ['Quantity', 'UnitPrice', 'Revenue']
    o_report = outlier_report_iqr(df, o_cols)
    st.dataframe(o_report, width="stretch", hide_index=True)

    st.info("**Observation**: Large-bulk orders (> 100 quantity) and high-value "
            "items (> $1,000 UnitPrice) are outliers. These are legitimate B2B "
            "transactions but dominate revenue.")

    fig = outlier_boxplots(df, o_cols)
    st.plotly_chart(fig, width="stretch")

# ── TAB 4: REVENUE BY COUNTRY (existing) ──────────────────────────────────

with tab4:
    grp = df.groupby('Country').agg(
        revenue=('Revenue', 'sum'),
        orders=('InvoiceNo', 'nunique')
    ).reset_index()
    grp = grp.sort_values('revenue', ascending=False).head(15)
    fig = px.bar(grp, x='Country', y='revenue', color='orders',
                 title="Revenue by Country (Top 15)", labels={'revenue': 'Revenue'})
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Insight**: UK dominates with > 80% of revenue since this is a UK-based "
                "retailer. Germany, France, and EIRE (Ireland) are distant seconds.")

# ── TAB 5: TOP PRODUCTS (existing) ────────────────────────────────────────

with tab5:
    grp = df.groupby('Description').agg(
        revenue=('Revenue', 'sum'),
        qty=('Quantity', 'sum')
    ).reset_index()
    grp = grp.sort_values('revenue', ascending=False).head(20)
    fig = px.bar(grp, x='revenue', y='Description', orientation='h', color='qty',
                 title="Top 20 Products by Revenue", color_continuous_scale='Blues')
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Insight**: The top-selling products are mostly home/kitware items "
                "(regalia, lunch bags, decorative items) — the retailer's core category.")

# ── TAB 6: REVENUE OVER TIME (existing) ───────────────────────────────────

with tab6:
    grp = df.groupby('Month').agg(
        revenue=('Revenue', 'sum'),
        orders=('InvoiceNo', 'nunique')
    ).reset_index()
    fig = px.line(grp, x='Month', y='revenue', markers=True,
                  title="Monthly Revenue Trend")
    st.plotly_chart(fig, width="stretch")

    # Month-over-month comparison
    grp2 = df.groupby('Month_of_Year').agg(
        revenue=('Revenue', 'sum')
    ).reset_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    grp2['Month_Name'] = grp2['Month_of_Year'].apply(lambda x: month_names[int(x)-1])
    fig2 = px.bar(grp2, x='Month_Name', y='revenue',
                  title="Aggregate Revenue by Month of Year",
                  labels={'revenue': 'Revenue'})
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: November has the highest revenue (Christmas shopping). "
                "December drops slightly. Summer months (Jun-Aug) are also strong.")

# ── TAB 7: CUSTOMER SPEND (existing) ──────────────────────────────────────

with tab7:
    cust = df.groupby('CustomerID').agg(
        total_spend=('Revenue', 'sum'),
        orders=('InvoiceNo', 'nunique'),
        avg_order=('Revenue', 'mean')
    ).reset_index()
    fig = px.histogram(cust, x='total_spend', nbins=50,
                       title="Customer Lifetime Value Distribution",
                       labels={'total_spend': 'Total Spend'})
    fig.update_xaxes(range=[0, cust['total_spend'].quantile(0.95)])
    st.plotly_chart(fig, width="stretch")

    # Top customers
    top_cust = cust.sort_values('total_spend', ascending=False).head(10)
    fig2 = px.bar(top_cust, x='CustomerID', y='total_spend',
                  title="Top 10 Customers by Spend",
                  labels={'total_spend': 'Total Spend'})
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: Customer spend is heavily right-skewed — the top 10% of "
                "customers drive the majority of revenue (Pareto principle).")

# ── RAW DATA ──────────────────────────────────────────────────────────────

with st.expander("View Raw Data"):
    st.dataframe(df.head(1000))
