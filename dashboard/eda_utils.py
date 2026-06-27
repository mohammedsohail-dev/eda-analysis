"""
EDA Utilities — shared data-quality, statistics, and outlier functions
used by every dashboard page.  Each function returns plotly figures or
pandas DataFrames ready for st.plotly_chart / st.dataframe.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ── 1. DATA QUALITY OVERVIEW ──────────────────────────────────────────────

def _arrow_safe_str(val):
    """Convert a value to a PyArrow-safe string representation."""
    if isinstance(val, pd.CategoricalDtype):
        return str(val)
    if hasattr(val, "name"):
        return val.name  # e.g. Int64Dtype -> "Int64"
    return str(val)


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with one row per column:
      - dtype, non-null count, null count, null %, cardinality, sample values
    """
    rows = []
    for col in df.columns:
        n_miss = df[col].isna().sum()
        # Safely attempt cardinality — list columns crash nunique()
        try:
            cardinality = df[col].nunique()
        except TypeError:
            cardinality = -1  # unhashable type (list, dict, etc.)
        sample_val = ""
        if cardinality > 0:
            try:
                sample_val = str(df[col].dropna().iloc[0])
            except (IndexError, TypeError):
                sample_val = "(unhashable)"
        rows.append({
            "Column": col,
            "Dtype": _arrow_safe_str(df[col].dtype),
            "Non-Null": int(df[col].notna().sum()),
            "Nulls": int(n_miss),
            "Null %": round(n_miss / len(df) * 100, 2),
            "Cardinality": cardinality if cardinality >= 0 else "N/A",
            "Sample": sample_val,
        })
    return pd.DataFrame(rows)


def duplicate_summary(df: pd.DataFrame) -> dict:
    """Return total rows, duplicate rows, and duplicate percentage."""
    # Exclude columns that contain unhashable types (list, dict, set, etc.)
    # which would crash df.duplicated()
    safe_cols = []
    for col in df.columns:
        # Check if any value in the column is a list/dict/set
        try:
            sample = df[col].dropna().iloc[0] if len(df) else None
            if isinstance(sample, (list, dict, set, tuple)):
                continue
            safe_cols.append(col)
        except (IndexError, TypeError):
            safe_cols.append(col)
    dups = df[safe_cols].duplicated().sum() if safe_cols else 0
    return {
        "Total Rows": len(df),
        "Duplicate Rows": dups,
        "Duplicate %": round(dups / len(df) * 100, 3) if len(df) else 0,
    }


def missing_value_chart(df: pd.DataFrame) -> go.Figure:
    """Plotly bar chart of missing-value percentage per column."""
    report = data_quality_report(df)
    report = report[report["Null %"] > 0].sort_values("Null %", ascending=False)
    if report.empty:
        # No missing values — return an empty figure with a text annotation
        fig = go.Figure()
        fig.add_annotation(text="No missing values in any column", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper")
        fig.update_layout(title="Missing Values", height=200)
        return fig
    fig = px.bar(report, x="Column", y="Null %",
                 title="Missing Values by Column (%)",
                 labels={"Null %": "Missing %"},
                 color="Null %", color_continuous_scale="Reds")
    fig.update_layout(xaxis_tickangle=-45)
    return fig


# ── 2. DESCRIPTIVE STATISTICS ─────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Return descriptive statistics (count, mean, std, min, 25%, 50%, 75%, max,
    skew, kurtosis) for the given numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter to columns that actually exist and are numeric
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return pd.DataFrame()
    desc = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["skew"] = df[cols].skew()
    desc["kurtosis"] = df[cols].kurtosis()
    desc = desc.rename(columns={
        "count": "Count", "mean": "Mean", "std": "Std", "min": "Min",
        "25%": "P25", "50%": "Median", "75%": "P75", "max": "Max",
        "skew": "Skew", "kurtosis": "Kurtosis",
    })
    # Round numeric columns for cleaner display
    for col in desc.select_dtypes(include=[np.number]).columns:
        desc[col] = desc[col].round(3)
    return desc


def distribution_histograms(df: pd.DataFrame, columns: list[str] | None = None,
                            bins: int = 40) -> go.Figure:
    """Plotly subplot histograms for each numeric column."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        fig = go.Figure()
        fig.add_annotation(text="No numeric columns to display", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    n = len(cols)
    fig = make_subplots(rows=n, cols=1, subplot_titles=cols, shared_xaxes=False,
                        vertical_spacing=0.08 / n * n if n > 1 else 0.1)
    for i, col in enumerate(cols, 1):
        fig.add_trace(go.Histogram(x=df[col].dropna(), nbinsx=bins, name=col,
                                   marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]),
                      row=i, col=1)
    fig.update_layout(height=200 * n, showlegend=False, title="Distribution Histograms")
    return fig


# ── 3. OUTLIER DETECTION (IQR method) ─────────────────────────────────────

def outlier_report_iqr(df: pd.DataFrame, columns: list[str] | None = None,
                       multiplier: float = 1.5) -> pd.DataFrame:
    """
    IQR-based outlier detection.  Returns a DataFrame with:
      column, outliers count, outlier %, lower bound, upper bound, min, max.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        rows.append({
            "Column": col,
            "Outliers": len(outliers),
            "Outlier %": round(len(outliers) / len(df) * 100, 2),
            "Lower Bound": round(lower, 3),
            "Upper Bound": round(upper, 3),
            "Min": round(df[col].min(), 3),
            "Max": round(df[col].max(), 3),
        })
    return pd.DataFrame(rows)


def outlier_boxplots(df: pd.DataFrame, columns: list[str] | None = None,
                     max_cols_per_row: int = 4) -> go.Figure:
    """
    Plotly box plots for numeric columns, grouped so that each row
    shows at most *max_cols_per_row* boxes.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        fig = go.Figure()
        fig.add_annotation(text="No numeric columns", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    n = len(cols)
    nrows = int(np.ceil(n / max_cols_per_row))
    fig = make_subplots(rows=nrows, cols=min(n, max_cols_per_row),
                        subplot_titles=cols,
                        horizontal_spacing=0.1, vertical_spacing=0.15)
    for i, col in enumerate(cols):
        r = i // max_cols_per_row + 1
        c = i % max_cols_per_row + 1
        fig.add_trace(go.Box(y=df[col].dropna(), name=col, boxmean="sd"),
                      row=r, col=c)
    fig.update_layout(height=300 * nrows, showlegend=False,
                      title="Outlier Box Plots (IQR method)")
    return fig


# ── 4. CORRELATION ANALYSIS ───────────────────────────────────────────────

def correlation_heatmap(df: pd.DataFrame, columns: list[str] | None = None,
                        method: str = "pearson") -> go.Figure:
    """
    Correlation heatmap for numeric columns.  Method can be
    'pearson', 'spearman', or 'kendall'.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 numeric columns for correlation",
                           showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    corr = df[cols].corr(method=method)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title=f"{method.title()} Correlation Matrix", aspect="auto",
                    zmin=-1, zmax=1)
    return fig


def top_correlations(df: pd.DataFrame, columns: list[str] | None = None,
                     n: int = 10, method: str = "pearson") -> pd.DataFrame:
    """Return the top-n absolute pairwise correlations."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        return pd.DataFrame()
    corr = df[cols].corr(method=method).unstack().reset_index()
    corr.columns = ["Var1", "Var2", "Correlation"]
    corr = corr[corr["Var1"] != corr["Var2"]]
    corr["Abs"] = corr["Correlation"].abs()
    corr = corr.sort_values("Abs", ascending=False).drop_duplicates(subset=["Abs"]).head(n)
    corr = corr.drop(columns=["Abs"]).reset_index(drop=True)
    return corr


# ── 5. CLASS IMBALANCE ────────────────────────────────────────────────────

def class_balance_chart(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Bar chart showing class distribution of a binary/categorical target."""
    grp = df[target_col].value_counts().reset_index()
    grp.columns = [target_col, "Count"]
    grp["Percent"] = (grp["Count"] / grp["Count"].sum() * 100).round(1)
    fig = px.bar(grp, x=target_col, y="Count",
                 text=grp["Count"].astype(str) + " (" + grp["Percent"].astype(str) + "%)",
                 title=f"Class Balance: {target_col}",
                 color=target_col, color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textposition="outside")
    return fig


# ── 6. HELPER: safely create subplots (avoid plotly import each time) ────

from plotly.subplots import make_subplots
