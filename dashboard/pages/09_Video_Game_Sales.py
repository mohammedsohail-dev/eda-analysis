import streamlit as st
import pandas as pd
import plotly.express as px
from eda_utils import (
    data_quality_report, duplicate_summary, missing_value_chart,
    descriptive_stats, outlier_report_iqr, outlier_boxplots,
    correlation_heatmap, top_correlations,
)

st.set_page_config(page_title="Video Game Sales", layout="wide")
st.title("Video Game Sales Analysis")
st.markdown("Global game sales by platform, genre & publisher (1980-2020, VGChartz).")

# ── LOAD DATA ─────────────────────────────────────────────────────────────

DATA = "../data/video-game-sales.csv.gz"
df = pd.read_csv(DATA)

# ── DATA CLEANING ─────────────────────────────────────────────────────────

# Drop rows missing critical fields
df = df.dropna(subset=['Year', 'Global_Sales']).copy()
df['Year'] = df['Year'].astype(int)

# Remove rows with zero global sales (data noise)
df = df[df['Global_Sales'] > 0].copy()

# Fill missing Publisher with 'Unknown'
df['Publisher'] = df['Publisher'].fillna('Unknown')

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────

# Platform generation classification
gen_map = {
    'PS1': 'Gen5', 'PS2': 'Gen6', 'PS3': 'Gen7', 'PS4': 'Gen8',
    'NES': 'Gen3', 'SNES': 'Gen4', 'N64': 'Gen5', 'GC': 'Gen6',
    'Wii': 'Gen7', 'WiiU': 'Gen8', 'Switch': 'Gen8',
    'GB': 'Gen4', 'GBA': 'Gen5', 'DS': 'Gen7', '3DS': 'Gen8',
    '2600': 'Gen2', 'GEN': 'Gen4', 'DC': 'Gen6', 'SAT': 'Gen5',
    'X360': 'Gen7', 'XB': 'Gen6', 'XOne': 'Gen8',
    'PSP': 'Gen7', 'PSV': 'Gen8',
    'PC': 'PC',
}
df['Generation'] = df['Platform'].map(gen_map).fillna('Other')

# Market share dominance: NA share of global
df['NA_Share'] = df['NA_Sales'] / df['Global_Sales']

# Regional preference: whether a game sells better in JP or NA
df['JP_vs_NA_Ratio'] = (df['JP_Sales'] + 1) / (df['NA_Sales'] + 1)

# Decade bucket
df['Decade'] = (df['Year'] // 10) * 10
df['Decade'] = df['Decade'].astype(str) + 's'

# Sales rank bracket
df['Sales_Tier'] = pd.cut(df['Global_Sales'],
                           bins=[0, 0.1, 0.5, 1, 5, 10, 100],
                           labels=['<0.1M', '0.1-0.5M', '0.5-1M', '1-5M', '5-10M', '10M+'])

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────

st.sidebar.header("Filters")
platforms = st.sidebar.multiselect("Platform", sorted(df['Platform'].unique()), default=None)
genres = st.sidebar.multiselect("Genre", sorted(df['Genre'].unique()), default=None)
year_range = st.sidebar.slider("Year Range", int(df['Year'].min()), int(df['Year'].max()), (2000, 2020))
if platforms:
    df = df[df['Platform'].isin(platforms)]
if genres:
    df = df[df['Genre'].isin(genres)]
df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

# ── KPI METRICS ───────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Games", f"{len(df):,}")
col2.metric("Global Sales", f"{df['Global_Sales'].sum():.0f}M")
col3.metric("Avg Sales/Game", f"{df['Global_Sales'].mean():.2f}M")
col4.metric("Platforms", f"{df['Platform'].nunique()}")
col5.metric("Publishers", f"{df['Publisher'].nunique():,}")

# ── TABS ──────────────────────────────────────────────────────────────────

tab_names = [
    " Data Quality", " Statistics", " Outliers",
    "Sales by Platform", "Genre Breakdown", "Sales Over Time", "Regional Split",
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
    num_cols = ['Year', 'Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    stats_df = descriptive_stats(df, num_cols)
    st.dataframe(stats_df, width="stretch")

    st.subheader("Global Sales Distribution")
    fig = px.histogram(df, x='Global_Sales', nbins=50,
                       title="Distribution of Global Game Sales",
                       labels={'Global_Sales': 'Global Sales (M)'})
    fig.update_xaxes(range=[0, df['Global_Sales'].quantile(0.95)])
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Insight**: Sales distribution is extremely right-skewed — most games "
                "sell less than 1M units, while a tiny fraction become 10M+ blockbusters.")

# ── TAB 3: OUTLIER ANALYSIS ───────────────────────────────────────────────

with tab3:
    st.subheader("Outlier Detection (IQR Method)")
    o_cols = ['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales']
    o_report = outlier_report_iqr(df, o_cols)
    st.dataframe(o_report, width="stretch", hide_index=True)

    st.info("**Observation**: Blockbuster games (> 10M global sales) are extreme outliers. "
            "These are the Wii Sports, GTA V, and Minecraft of the world — "
            "cultural phenomena rather than typical game performance.")

    fig = outlier_boxplots(df, o_cols)
    st.plotly_chart(fig, width="stretch")

# ── TAB 4: SALES BY PLATFORM (existing) ───────────────────────────────────

with tab4:
    grp = df.groupby('Platform').agg(
        sales=('Global_Sales', 'sum'),
        games=('Name', 'count')
    ).reset_index()
    grp = grp.sort_values('sales', ascending=False).head(15)
    fig = px.bar(grp, x='Platform', y='sales', color='games',
                 color_continuous_scale='Blues',
                 title="Global Sales by Platform (Top 15)",
                 labels={'sales': 'Sales (M)'})
    st.plotly_chart(fig, width="stretch")

    # By generation
    grp2 = df.groupby('Generation').agg(
        sales=('Global_Sales', 'sum'),
        games=('Name', 'count')
    ).reset_index()
    grp2 = grp2.sort_values('sales', ascending=False)
    fig2 = px.bar(grp2, x='Generation', y='sales', color='games',
                  color_continuous_scale='Viridis',
                  title="Global Sales by Console Generation",
                  labels={'sales': 'Sales (M)'})
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: PS2 leads in total sales, driven by its massive install base. "
                "Gen7 (PS3/X360/Wii) was the peak generation for software revenue.")

# ── TAB 5: GENRE BREAKDOWN (existing) ─────────────────────────────────────

with tab5:
    grp = df.groupby('Genre').agg(
        sales=('Global_Sales', 'sum'),
        games=('Name', 'count')
    ).reset_index()
    fig = px.pie(grp, values='sales', names='Genre',
                 title="Global Sales by Genre")
    st.plotly_chart(fig, width="stretch")

    # Genre × Generation heatmap
    grp2 = df.groupby(['Generation', 'Genre']).agg(
        sales=('Global_Sales', 'sum')
    ).reset_index()
    pivot = grp2.pivot_table(index='Generation', columns='Genre', values='sales', aggfunc='sum')
    fig2 = px.imshow(pivot, text_auto='.0f',
                     color_continuous_scale='Blues',
                     title="Genre Sales by Generation (M units)",
                     aspect='auto')
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: Action is the highest-grossing genre overall. "
                "Shooter and Sports genres grew significantly in later generations.")

# ── TAB 6: SALES OVER TIME (existing) ─────────────────────────────────────

with tab6:
    grp = df.groupby('Year').agg(
        sales=('Global_Sales', 'sum'),
        games=('Name', 'count')
    ).reset_index()
    fig = px.line(grp, x='Year', y='sales', markers=True,
                  title="Global Sales Over Time",
                  labels={'sales': 'Sales (M)'})
    fig.update_traces(line=dict(width=3), marker=dict(size=6))
    st.plotly_chart(fig, width="stretch")

    # Number of games released per year
    fig2 = px.bar(grp, x='Year', y='games',
                  title="Games Released per Year",
                  labels={'games': 'Games Released'})
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: The market peaked around 2008-2011 (PS3/X360 era). "
                "After 2015, the number of reported titles declined as the industry "
                "consolidated toward high-budget blockbusters.")

# ── TAB 7: REGIONAL SPLIT (existing) ──────────────────────────────────────

with tab7:
    region_df = df[['Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].melt(
        id_vars=['Name'], var_name='Region', value_name='Sales'
    )
    region_df['Region'] = region_df['Region'].str.replace('_Sales', '')
    grp = region_df.groupby('Region').agg(sales=('Sales', 'sum')).reset_index()
    fig = px.bar(grp, x='Region', y='sales', color='Region',
                 title="Sales by Region",
                 labels={'sales': 'Sales (M)'})
    st.plotly_chart(fig, width="stretch")

    # Regional preference by genre
    genre_region = df.groupby('Genre').agg(
        NA=('NA_Sales', 'sum'),
        EU=('EU_Sales', 'sum'),
        JP=('JP_Sales', 'sum'),
    ).reset_index()
    genre_region_melted = genre_region.melt(id_vars=['Genre'], var_name='Region', value_name='Sales')
    fig2 = px.bar(genre_region_melted, x='Genre', y='Sales', color='Region', barmode='group',
                  title="Sales by Genre and Region",
                  labels={'Sales': 'Sales (M)'})
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: NA is the largest market across all genres. "
                "Japan has a strong preference for Role-Playing games, while "
                "NA/EU favor Action and Shooter genres.")

# ── RAW DATA ──────────────────────────────────────────────────────────────

with st.expander("View Raw Data"):
    st.dataframe(df.head(1000))
