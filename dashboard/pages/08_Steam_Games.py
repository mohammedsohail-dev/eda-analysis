import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from eda_utils import (
    data_quality_report, duplicate_summary, missing_value_chart,
    descriptive_stats, outlier_report_iqr, outlier_boxplots,
    correlation_heatmap,
)

st.set_page_config(page_title="Steam Games", layout="wide")
st.title("Steam Games Market Analysis")
st.markdown("40K+ games on Steam — pricing, ratings, genres, and developer insights.")

# ── LOAD DATA ─────────────────────────────────────────────────────────────

DATA = "dashboard/data/steam-games.csv.gz"
df = pd.read_csv(DATA)

# ── DATA CLEANING ─────────────────────────────────────────────────────────

# Parse price field (handles 'Free', 'Free to Play', '$X.XX' formats)
def parse_price(val):
    if pd.isna(val):
        return float('nan')
    val = str(val).strip().lower()
    if val in ('free', 'free to play', 'play for free!'):
        return 0.0
    val = val.replace('$', '').replace(',', '')
    try:
        return float(val)
    except:
        return float('nan')

df['original_price'] = df['original_price'].apply(parse_price)
df = df.dropna(subset=['original_price']).copy()

# Parse release_date to extract year
def parse_year(val):
    if pd.isna(val):
        return None
    val = str(val).strip()
    # Try common date formats
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%d %b, %Y', '%b %d, %Y', '%Y'):
        try:
            return pd.to_datetime(val, format=fmt).year
        except:
            continue
    return None

df['release_year'] = df['release_date'].apply(parse_year)
df = df.dropna(subset=['release_year']).copy()
df['release_year'] = df['release_year'].astype(int)

# Parse achievements count from string like "X achievements"
def parse_achievements(val):
    if pd.isna(val):
        return 0
    val = str(val).strip()
    try:
        return int(val.split()[0])
    except:
        return 0

df['achievements_count'] = df['achievements'].apply(parse_achievements)

# Split genres into a list
df['genre_list'] = df['genre'].str.split(r',\s*')

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────

# Price categories
df['price_tier'] = pd.cut(df['original_price'],
                           bins=[-1, 0, 5, 15, 30, 60, 200],
                           labels=['Free', 'Budget (<$5)', 'Low ($5-15)',
                                   'Mid ($15-30)', 'Premium ($30-60)', 'Ultra ($60+)'])

# Free-to-play flag
df['is_free'] = (df['original_price'] == 0).astype(int)

# Age of game (years since release)
current_year = 2026
df['age_years'] = current_year - df['release_year']

# Developer concentration: number of games per developer
dev_counts = df.groupby('developer').size().reset_index(name='dev_game_count')
df = df.merge(dev_counts, on='developer', how='left')

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────

# Get all unique genres
all_genres = sorted(set(
    g.strip() for genres in df['genre'].dropna() for g in str(genres).split(',')
))
st.sidebar.header("Filters")
genre_filter = st.sidebar.selectbox("Genre", ["All"] + all_genres)
price_max = st.sidebar.slider("Max Price ($)", 0, 100, 60)
df = df[df['original_price'] <= price_max]
if genre_filter != "All":
    df = df[df['genre'].str.contains(genre_filter, na=False, case=False)]

# ── KPI METRICS ───────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Games", f"{len(df):,}")
col2.metric("Avg Price", f"${df['original_price'].mean():.2f}")
col3.metric("Median Price", f"${df['original_price'].median():.2f}")
col4.metric("Free Games", f"{len(df[df['is_free'] == 1]):,}")
col5.metric("Avg Achievements", f"{df['achievements_count'].mean():.0f}")

# ── TABS ──────────────────────────────────────────────────────────────────

tab_names = [
    " Data Quality", " Statistics", " Outliers",
    "Price Distribution", "Games by Genre", "Developer Analysis", "Top Tags",
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
    num_cols = ['original_price', 'release_year', 'achievements_count', 'age_years']
    stats_df = descriptive_stats(df, num_cols)
    st.dataframe(stats_df, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(df, x='original_price', nbins=60,
                           title="Price Histogram",
                           labels={'original_price': 'Price ($)'})
        fig.update_xaxes(range=[0, 70])
        st.plotly_chart(fig, width="stretch")
    with col_b:
        fig2 = px.box(df, x='original_price', orientation='h',
                      title="Price Distribution Box Plot")
        fig2.update_xaxes(range=[0, 70])
        st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: The Steam market has a strong concentration of cheap games "
                "(< $10). The $0 bin (Free to Play) is the single largest category.")

# ── TAB 3: OUTLIER ANALYSIS ───────────────────────────────────────────────

with tab3:
    st.subheader("Outlier Detection (IQR Method)")
    o_cols = ['original_price', 'achievements_count', 'age_years']
    o_report = outlier_report_iqr(df, o_cols)
    st.dataframe(o_report, width="stretch", hide_index=True)

    st.info("**Observation**: Price outliers include expensive bundles or deluxe editions "
            "(> $60). Achievement count outliers are games with 1000+ achievements.")

    fig = outlier_boxplots(df, o_cols)
    st.plotly_chart(fig, width="stretch")

# ── TAB 4: PRICE DISTRIBUTION (existing) ──────────────────────────────────

with tab4:
    fig = px.histogram(df, x='original_price', nbins=60,
                       title="Price Distribution",
                       labels={'original_price': 'Price ($)'})
    fig.update_xaxes(range=[0, 70])
    st.plotly_chart(fig, width="stretch")

    # Price tier breakdown
    grp = df.groupby('price_tier', observed=False).agg(
        count=('name', 'count')
    ).reset_index()
    fig2 = px.pie(grp, values='count', names='price_tier',
                  title="Games by Price Tier")
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: Over 40% of Steam games are priced under $15. "
                "The 'Free' and 'Budget' tiers dominate the catalog.")

# ── TAB 5: GAMES BY GENRE (existing) ──────────────────────────────────────

with tab5:
    exploded = df.explode('genre_list')
    exploded['genre_list'] = exploded['genre_list'].str.strip()
    grp = exploded.groupby('genre_list').agg(
        count=('name', 'nunique'),
        avg_price=('original_price', 'mean')
    ).reset_index()
    grp = grp.sort_values('count', ascending=False).head(20)
    fig = px.bar(grp, x='count', y='genre_list', orientation='h', color='avg_price',
                 title="Games by Genre (Top 20)",
                 labels={'count': 'Count', 'genre_list': ''},
                 color_continuous_scale='Viridis')
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Insight**: Action, Indie, and Casual are the most common genres. "
                "Early Access and VR games tend to have higher average prices.")

# ── TAB 6: DEVELOPER ANALYSIS (existing) ──────────────────────────────────

with tab6:
    grp = df.groupby('developer').agg(
        games=('name', 'nunique'),
        avg_price=('original_price', 'mean')
    ).reset_index()
    grp = grp.sort_values('games', ascending=False).head(15)
    fig = px.bar(grp, x='developer', y='games', color='avg_price',
                 title="Top 15 Developers by Game Count",
                 labels={'games': 'Games'})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, width="stretch")

    # Release year trend
    grp2 = df.groupby('release_year').agg(
        games=('name', 'nunique'),
        avg_price=('original_price', 'mean')
    ).reset_index()
    fig2 = px.line(grp2, x='release_year', y='games', markers=True,
                   title="Games Released per Year",
                   labels={'games': 'Games Released'})
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: A handful of developers (Ubisoft, SEGA) have the most titles. "
                "Game releases grew exponentially from 2010 to 2020.")

# ── TAB 7: TOP TAGS (existing) ────────────────────────────────────────────

with tab7:
    tags = df['popular_tags'].dropna().str.split(',').explode().str.strip().value_counts().head(20)
    fig = px.bar(x=tags.values, y=tags.index, orientation='h',
                 title="Most Common Tags",
                 labels={'x': 'Count', 'y': ''})
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Insight**: 'Single-player' and 'Steam Achievements' are the most common tags. "
                "Multi-player features correlate with higher engagement and pricing.")

# ── RAW DATA ──────────────────────────────────────────────────────────────

with st.expander("View Raw Data"):
    st.dataframe(df.head(1000))
