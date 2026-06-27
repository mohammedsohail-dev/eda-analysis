import streamlit as st
import pandas as pd
import plotly.express as px
from eda_utils import (
    data_quality_report, duplicate_summary, missing_value_chart,
    descriptive_stats, outlier_report_iqr, outlier_boxplots,
    correlation_heatmap,
)

st.set_page_config(page_title="NYC Taxi Trips", layout="wide")
st.title("NYC Yellow Taxi Trip Analysis")
st.markdown("Trip patterns, tipping behavior, and fare analysis (Jan 2024, 100k sample).")

# ── LOAD DATA ─────────────────────────────────────────────────────────────

DATA = "dashboard/data/nyc-taxi-trips.csv.gz"
df = pd.read_csv(DATA, parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

# ── DATA CLEANING ─────────────────────────────────────────────────────────

# Remove zero/negative fares and distances
df = df[df['fare_amount'] > 0].copy()
df = df[df['trip_distance'] > 0].copy()
df = df[df['passenger_count'] > 0].copy()

# Remove trips with zero total amount (likely data errors)
df = df[df['total_amount'] > 0].copy()

# Cap tip percentage at 100% to remove extreme outliers
df['tip_pct'] = (df['tip_amount'] / df['fare_amount'] * 100)
df = df[df['tip_pct'] <= 100].copy()

# ── FEATURE ENGINEERING ────────────────────────────────────────────────────

# Extract temporal features
df['hour'] = df['tpep_pickup_datetime'].dt.hour
df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek  # 0=Monday
df['weekend'] = (df['day_of_week'] >= 5).astype(int)
df['month_day'] = df['tpep_pickup_datetime'].dt.day

# Trip duration in minutes
df['trip_duration_min'] = (
    df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
).dt.total_seconds() / 60

# Remove impossibly long or short trips (> 6 hours or < 1 minute)
df = df[(df['trip_duration_min'] >= 1) & (df['trip_duration_min'] <= 360)].copy()

# Average speed (mph)
df['speed_mph'] = df['trip_distance'] / (df['trip_duration_min'] / 60)

# Remove unrealistic speeds (> 80 mph in NYC)
df = df[df['speed_mph'] <= 80].copy()

# Fare per mile
df['fare_per_mile'] = df['fare_amount'] / df['trip_distance']

# Payment type labels
payment_map = {1: 'Credit Card', 2: 'Cash', 3: 'No Charge', 4: 'Dispute', 5: 'Unknown'}
df['payment_label'] = df['payment_type'].map(payment_map).fillna('Other')

# Tip flag
df['tipped'] = (df['tip_amount'] > 0).astype(int)

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────

st.sidebar.header("Filters")
min_dist = st.sidebar.slider("Min Trip Distance (miles)", 0.0, 20.0, 0.0)
max_fare = st.sidebar.slider("Max Fare ($)", 0, 200, 200)
df = df[(df['trip_distance'] >= min_dist) & (df['fare_amount'] <= max_fare)]

# ── KPI METRICS ───────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Trips", f"{len(df):,}")
col2.metric("Avg Fare", f"${df['fare_amount'].mean():.2f}")
col3.metric("Avg Tip", f"${df['tip_amount'].mean():.2f}")
col4.metric("Avg Distance", f"{df['trip_distance'].mean():.1f} mi")
col5.metric("Tip Rate", f"{df['tipped'].mean():.1%}")

# ── TABS ──────────────────────────────────────────────────────────────────

tab_names = [
    " Data Quality", " Statistics", " Outliers",
    "Trips by Hour", "Fare vs Distance", "Tip Analysis", "Passenger Count",
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

# ── TAB 2: STATISTICAL SUMMARY ────────────────────────────────────────────

with tab2:
    st.subheader("Descriptive Statistics")
    num_cols = ['fare_amount', 'tip_amount', 'trip_distance', 'trip_duration_min',
                'speed_mph', 'passenger_count', 'total_amount']
    stats_df = descriptive_stats(df, num_cols)
    st.dataframe(stats_df, width="stretch")

    st.subheader("Distributions")
    selected = st.multiselect("Select variables", num_cols,
                               default=['fare_amount', 'trip_distance', 'trip_duration_min'])
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
    o_cols = ['fare_amount', 'tip_amount', 'trip_distance', 'trip_duration_min', 'speed_mph']
    o_report = outlier_report_iqr(df, o_cols)
    st.dataframe(o_report, width="stretch", hide_index=True)

    st.info("**Observation**: `trip_distance` and `fare_amount` have high-end outliers "
            "(rides to/from airports, long-haul trips). These are legitimate but "
            "should be analyzed separately from typical city rides.")

    fig = outlier_boxplots(df, o_cols)
    st.plotly_chart(fig, width="stretch")

# ── TAB 4: TRIPS BY HOUR (existing) ───────────────────────────────────────

with tab4:
    grp = df.groupby('hour').agg(
        trips=('fare_amount', 'count'),
        avg_fare=('fare_amount', 'mean')
    ).reset_index()
    fig = px.bar(grp, x='hour', y='trips', color='avg_fare',
                 color_continuous_scale='Viridis',
                 title="Trips by Hour of Day", labels={'trips': 'Trip Count'})
    st.plotly_chart(fig, width="stretch")

    # Hourly average fare
    fig2 = px.line(grp, x='hour', y='avg_fare', markers=True,
                   title="Average Fare by Hour",
                   labels={'avg_fare': 'Avg Fare ($)'})
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: Peak demand at 6 PM. Fares are highest late night (2-4 AM) "
                "when longer airport trips are more common.")

# ── TAB 5: FARE VS DISTANCE (existing) ────────────────────────────────────

with tab5:
    fig = px.scatter(df.sample(5000), x='trip_distance', y='fare_amount', color='tip_amount',
                     title="Fare vs Distance (sample 5k)",
                     labels={'trip_distance': 'Distance (mi)'},
                     color_continuous_scale='Viridis')
    st.plotly_chart(fig, width="stretch")

    # Fare per mile by distance bin
    df['dist_bin'] = pd.cut(df['trip_distance'], bins=[0, 1, 2, 5, 10, 50],
                             labels=['<1', '1-2', '2-5', '5-10', '10+'])
    grp = df.groupby('dist_bin', observed=False).agg(
        fare_per_mile=('fare_per_mile', 'mean')
    ).reset_index()
    fig2 = px.bar(grp, x='dist_bin', y='fare_per_mile',
                  title="Fare per Mile by Distance Bucket",
                  labels={'dist_bin': 'Distance (mi)', 'fare_per_mile': '$/mi'})
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: Short trips have the highest $/mile due to the base fare — "
                "longer trips are more efficient per mile.")

# ── TAB 6: TIP ANALYSIS (existing) ────────────────────────────────────────

with tab6:
    fig = px.histogram(df, x='tip_pct', nbins=50,
                       title="Tip Percentage Distribution",
                       labels={'tip_pct': 'Tip %'})
    fig.update_xaxes(range=[0, 50])
    st.plotly_chart(fig, width="stretch")

    # Tip rate by payment type
    grp = df.groupby('payment_label').agg(
        tip_rate=('tipped', 'mean'),
        avg_tip=('tip_amount', 'mean'),
        count=('tipped', 'count')
    ).reset_index()
    fig2 = px.bar(grp, x='payment_label', y='tip_rate', color='avg_tip',
                  color_continuous_scale='Greens',
                  title="Tip Rate by Payment Type",
                  labels={'payment_label': 'Payment', 'tip_rate': 'Tip Rate'})
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: Nearly all credit card trips include a tip (pre-selected options). "
                "Cash tips are rare since they're harder to track in the data.")

# ── TAB 7: PASSENGER COUNT (existing) ─────────────────────────────────────

with tab7:
    grp = df.groupby('passenger_count').agg(
        trips=('fare_amount', 'count'),
        avg_fare=('fare_amount', 'mean')
    ).reset_index()
    fig = px.pie(grp, values='trips', names='passenger_count',
                 title="Trips by Passenger Count")
    st.plotly_chart(fig, width="stretch")

    # Weekend vs Weekday comparison
    grp2 = df.groupby('weekend').agg(trips=('fare_amount', 'count')).reset_index()
    grp2['weekend'] = grp2['weekend'].map({0: 'Weekday', 1: 'Weekend'})
    fig2 = px.bar(grp2, x='weekend', y='trips', color='weekend',
                  title="Weekday vs Weekend Trips")
    st.plotly_chart(fig2, width="stretch")

    st.markdown("**Insight**: Single-passenger trips dominate (70%+). "
                "Weekend trips have slightly higher average fares.")

# ── RAW DATA ──────────────────────────────────────────────────────────────

with st.expander("View Raw Data"):
    st.dataframe(df.head(1000))
