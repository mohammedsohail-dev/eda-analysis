import streamlit as st

st.set_page_config(
    page_title="Data Analyst Portfolio",
    page_icon="",
    layout="wide",
)

st.title("Data Analyst Portfolio")
st.markdown("Interactive dashboards for banking, finance, insurance & gaming.")

projects = [
    ("Bank Marketing", "Predict term deposit subscriptions. Campaign analytics, demographics & conversion funnel.", "01_Bank_Marketing", "41,188 rows"),
    ("Credit Card Fraud", "Fraud detection analysis. Imbalanced classification, transaction patterns.", "02_Credit_Card_Fraud", "284,807 rows"),
    ("Loan Default Risk", "Credit risk analysis. Loan approval, FICO, debt-to-income ratios.", "03_Loan_Default_Risk", "10,000 rows"),
    ("Medical Cost Insurance", "Insurance cost drivers. Age, BMI, smoking impact on premiums.", "04_Medical_Cost_Insurance", "1,338 rows"),
    ("NYC Taxi Trips", "Trip patterns & tipping. Geospatial & temporal ride analysis.", "05_NYC_Taxi_Trips", "100,000 rows"),
    ("Online Retail Sales", "Customer & product analytics. Sales trends, CLV, country breakdown.", "06_Online_Retail_Sales", "541,909 rows"),
    ("S&P 500 Bank Stocks", "Stock performance & correlation. 5-year price trends for top banks.", "07_SP500_Bank_Stocks", "11,286 rows"),
    ("Steam Games", "Gaming market analysis. Pricing, ratings, genres & developers.", "08_Steam_Games", "40,833 rows"),
    ("Video Game Sales", "Global game sales by platform, genre & publisher. Regional breakdowns.", "09_Video_Game_Sales", "16,598 rows"),
]

cols = st.columns(3)
for i, (title, desc, page, rows) in enumerate(projects):
    with cols[i % 3]:
        st.markdown(f"### {title}")
        st.caption(f"_{rows}_")
        st.markdown(desc)
        if st.button(f"Open →", key=page):
            st.switch_page(f"pages/{page}.py")
        st.divider()
