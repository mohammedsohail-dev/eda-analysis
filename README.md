# EDA Dashboard — 9 Real-World Datasets

An interactive Streamlit dashboard for exploratory data analysis across 9 diverse datasets covering banking, finance, fraud detection, insurance, transportation, retail, and gaming.

**Live demo**: [share.streamlit.io](https://share.streamlit.io) (deploy from this repo)

## Datasets

| # | Dataset | Size | Target |
|---|---------|------|--------|
| 1 | **Bank Marketing** | 41,188 rows | Term deposit subscription prediction |
| 2 | **Credit Card Fraud** | 284,807 rows | Fraudulent transaction detection |
| 3 | **Loan Default Risk** | 10,000 rows | Loan default prediction |
| 4 | **Medical Cost Insurance** | 1,338 rows | Premium cost drivers |
| 5 | **NYC Taxi Trips** | 100,000 rows | Trip fare & tipping analysis |
| 6 | **Online Retail Sales** | 541,909 rows | Revenue & customer analytics |
| 7 | **S&P 500 Bank Stocks** | 11,286 rows | Stock price & correlation analysis |
| 8 | **Steam Games** | 40,833 rows | Game pricing & ratings |
| 9 | **Video Game Sales** | 16,598 rows | Global sales by platform & genre |

## Features

- **Data Quality** — missing values, cardinality, duplicate detection, dtype overview
- **Statistical Summary** — descriptive stats, distribution histograms, correlation matrices
- **Outlier Analysis** — IQR-based detection, box plots, outlier percentages
- **Class Balance** — target variable distribution (binary classification datasets)
- **Feature Engineering** — derived features per dataset (age groups, loan-to-income, tip percentage, price tiers, etc.)
- **Key Insights** — data-driven findings for each domain

## Tech Stack

- **Python 3.14** — Pandas, NumPy, Plotly
- **Streamlit 1.58** — interactive multi-page dashboard
- **EDA Utilities** — shared module for data quality reports, statistics, outlier detection, and correlation analysis

## Local Setup

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/Dashboard_Home.py
```

## Deployment

This repo is ready for [Streamlit Community Cloud](https://share.streamlit.io). Point the main file to `dashboard/Dashboard_Home.py`.

## Project Structure

```
.
├── README.md
├── dashboard/
│   ├── Dashboard_Home.py       # Main entry point
│   ├── eda_utils.py            # Shared EDA utilities
│   ├── requirements.txt
│   ├── pages/                  # One page per dataset
│   └── data/                   # CSV datasets (gzip compressed)
└── reports/                    # Generated markdown reports
```

## Insights Highlights

- **Bank Marketing**: Conversion rate ~11%, key drivers are last contact duration and Euribor 3-month rate
- **Credit Card Fraud**: Only 0.17% of transactions are fraudulent — extreme class imbalance
- **Loan Default**: Grade G loans default at 3x the rate of Grade A loans
- **Medical Insurance**: Smokers pay 3-4x more in premiums than non-smokers
- **NYC Taxi**: 72% of trips are paid by credit card; average tip is $3.62
- **Online Retail**: UK accounts for 80%+ of revenue; October-November are peak months
- **Bank Stocks**: JPM and C have the highest intraday volatility among the 9 tickers
- **Steam Games**: Free-to-play games dominate player counts; price correlates weakly with rating
- **Video Game Sales**: North America leads at ~41% of global sales; PS2 is the best-selling platform
