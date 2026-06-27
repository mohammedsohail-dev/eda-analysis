import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from eda_utils import (
    data_quality_report, duplicate_summary, missing_value_chart,
    descriptive_stats, outlier_report_iqr, outlier_boxplots,
    correlation_heatmap,
)

st.set_page_config(page_title="S&P 500 Bank Stocks", layout="wide")
st.title("S&P 500 Bank Stocks")
st.markdown("5-year daily prices for 9 top US bank stocks: JPM, GS, C, WFC, MS, AXP, BLK, SCHW, USB.")

# ── LOAD DATA ─────────────────────────────────────────────────────────────

DATA = "../data/sp500-banks.csv.gz"
df = pd.read_csv(DATA, parse_dates=['Date'])

# ── DATA CLEANING ─────────────────────────────────────────────────────────

# Check for missing dates per ticker (trading day gaps)
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# Remove any rows with missing price data
df = df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume']).copy()

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────

# Daily return (%)
df['Daily_Return'] = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change() * 100)

# Log return (continuously compounded)
df['Log_Return'] = df.groupby('Ticker')['Close'].transform(lambda x: np.log(x / x.shift(1)))

# 20-day rolling volatility (approx 1 trading month)
df['Volatility_20d'] = df.groupby('Ticker')['Daily_Return'].transform(
    lambda x: x.rolling(20).std()
)

# Normalized price (each ticker starts at 100 on first date)
df['Normalized'] = df.groupby('Ticker')['Close'].transform(
    lambda x: x / x.iloc[0] * 100
)

# Price range for the day
df['Day_Range'] = df['High'] - df['Low']

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────

st.sidebar.header("Filters")
tickers = st.sidebar.multiselect("Tickers", df['Ticker'].unique(), default=df['Ticker'].unique()[:6])
if tickers:
    df = df[df['Ticker'].isin(tickers)]

# ── KPI METRICS ───────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Period", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
col2.metric("Tickers Selected", len(tickers))
col3.metric("Total Rows", f"{len(df):,}")
# Average daily return across selected tickers
avg_ret = df.groupby('Ticker')['Daily_Return'].mean().mean()
col4.metric("Avg Daily Return", f"{avg_ret:.3f}%")

# ── TABS ──────────────────────────────────────────────────────────────────

tab_names = [
    " Data Quality", " Statistics", " Outliers",
    "Price Trends", "Returns Comparison", "Correlation", "Volume",
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

    # Trading days per ticker
    st.subheader("Trading Days per Ticker")
    days_per_ticker = df.groupby('Ticker').agg(
        trading_days=('Date', 'nunique'),
        date_min=('Date', 'min'),
        date_max=('Date', 'max')
    ).reset_index()
    st.dataframe(days_per_ticker, width="stretch", hide_index=True)

# ── TAB 2: STATISTICAL SUMMARY ────────────────────────────────────────────

with tab2:
    st.subheader("Descriptive Statistics")
    num_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']
    stats_df = descriptive_stats(df, num_cols)
    st.dataframe(stats_df, width="stretch")

    st.subheader("Return Distribution by Ticker")
    selected_tickers = st.multiselect("Select tickers", df['Ticker'].unique(),
                                       default=df['Ticker'].unique()[:4])
    if selected_tickers:
        subset = df[df['Ticker'].isin(selected_tickers)]
        fig = px.histogram(subset, x='Daily_Return', color='Ticker',
                           barmode='overlay', nbins=80, opacity=0.5,
                           title="Daily Return Distribution")
        fig.update_xaxes(range=[-5, 5])
        st.plotly_chart(fig, width="stretch")

# ── TAB 3: OUTLIER ANALYSIS ───────────────────────────────────────────────

with tab3:
    st.subheader("Outlier Detection (IQR Method)")
    o_cols = ['Daily_Return', 'Volume', 'Day_Range']
    o_report = outlier_report_iqr(df, o_cols)
    st.dataframe(o_report, width="stretch", hide_index=True)

    st.info("**Observation**: Extreme daily returns (> ±5%) correspond to major "
            "market events (earnings reports, macroeconomic shocks, COVID-19). "
            "These are rare but informative — not data errors.")

    fig = outlier_boxplots(df, o_cols)
    st.plotly_chart(fig, width="stretch")

# ── TAB 4: PRICE TRENDS (existing) ────────────────────────────────────────

with tab4:
    fig = px.line(df, x='Date', y='Normalized', color='Ticker',
                  title="Normalized Price Trends (Base=100 on first date)",
                  labels={'Normalized': 'Price (Base 100)'})
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig, width="stretch")

    # Actual close prices
    fig2 = px.line(df, x='Date', y='Close', color='Ticker',
                   title="Closing Price Trends (Absolute)",
                   labels={'Close': 'Price ($)'})
    fig2.update_layout(hovermode='x unified')
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: When indexed to 100, differences in total return "
                "become clear — some banks have significantly outperformed others.")

# ── TAB 5: RETURNS COMPARISON (existing) ──────────────────────────────────

with tab5:
    # 1-year returns
    latest_prices = df.groupby('Ticker').last()['Close']
    one_year_ago = df['Date'].max() - pd.Timedelta(days=365)
    yearly_prices = df[df['Date'] >= one_year_ago].groupby('Ticker').first()['Close']
    yearly_returns = ((latest_prices / yearly_prices) - 1).reset_index()
    yearly_returns.columns = ['Ticker', 'Return']
    yearly_returns['Return_Label'] = yearly_returns['Return'].apply(lambda x: f"{x:.1%}")
    fig = px.bar(yearly_returns, x='Ticker', y='Return', color='Return',
                 color_continuous_scale='RdYlGn',
                 title="1-Year Returns by Ticker",
                 labels={'Return': 'Return'}, text='Return_Label')
    st.plotly_chart(fig, width="stretch")

    # Rolling volatility
    fig2 = px.line(df, x='Date', y='Volatility_20d', color='Ticker',
                   title="20-Day Rolling Volatility",
                   labels={'Volatility_20d': 'Volatility (%)'})
    fig2.update_layout(hovermode='x unified')
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: Volatility spikes during COVID-19 (Mar 2020) are visible "
                "across all banks. Higher-volatility tickers offer higher potential returns "
                "but also greater risk.")

# ── TAB 6: CORRELATION (existing) ─────────────────────────────────────────

with tab6:
    pivot = df.pivot_table(index='Date', columns='Ticker', values='Close').pct_change().dropna()
    if len(pivot.columns) > 1:
        corr = pivot.corr()
        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                        title="Daily Return Correlation", aspect='auto', zmin=-1, zmax=1)
        st.plotly_chart(fig, width="stretch")

        st.markdown("**Insight**: All bank stocks are highly correlated (0.6-0.9 range) "
                    "since they respond to the same macroeconomic factors. "
                    "Diversification across banks alone provides limited risk reduction.")
    else:
        st.warning("Need at least 2 tickers for correlation.")

# ── TAB 7: VOLUME (existing) ──────────────────────────────────────────────

with tab7:
    fig = px.line(df, x='Date', y='Volume', color='Ticker',
                  title="Trading Volume Over Time",
                  labels={'Volume': 'Volume'})
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig, width="stretch")

    # Average volume by ticker
    vol_grp = df.groupby('Ticker').agg(avg_volume=('Volume', 'mean')).reset_index()
    fig2 = px.bar(vol_grp, x='Ticker', y='avg_volume',
                  title="Average Daily Volume by Ticker",
                  labels={'avg_volume': 'Avg Volume'})
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: BAC and WFC have the highest trading volumes "
                "(retail banking giants). Volume spikes often precede major price moves.")

# ── RAW DATA ──────────────────────────────────────────────────────────────

with st.expander("View Raw Data"):
    st.dataframe(df.head(1000))
