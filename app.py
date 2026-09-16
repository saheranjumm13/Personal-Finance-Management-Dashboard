
# =====================================================
# IMPORT LIBRARIES
# =====================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from utils import *

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="💰 Personal Finance Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD CSS
# =====================================================

with open("style.css") as css:
    st.markdown(
        f"<style>{css.read()}</style>",
        unsafe_allow_html=True
    )

# =====================================================
# LOAD DATA
# =====================================================

# NOTE:
# If your Streamlit version is old,
# DO NOT use @st.cache_data

def load_data():

    df = pd.read_csv("personal_transactions.csv")

    df["Date"] = pd.to_datetime(df["Date"])

    return df


df = load_data()

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("personal_transactions.pkl")

# Load Label Encoders
category_encoder = joblib.load("category_encoder.pkl")
account_encoder = joblib.load("account_encoder.pkl")
transaction_encoder = joblib.load("transaction_encoder.pkl")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2489/2489756.png",
    width=120
)

st.sidebar.title("💰 Finance Dashboard")

st.sidebar.markdown("---")

# =====================================================
# FILTERS
# =====================================================

category = st.sidebar.multiselect(
    "📂 Category",
    options=sorted(df["Category"].unique()),
    default=sorted(df["Category"].unique())
)

transaction = st.sidebar.multiselect(
    "💳 Transaction Type",
    options=sorted(df["Transaction Type"].unique()),
    default=sorted(df["Transaction Type"].unique())
)

account = st.sidebar.multiselect(
    "🏦 Account Name",
    options=sorted(df["Account Name"].unique()),
    default=sorted(df["Account Name"].unique())
)

# =====================================================
# DATE FILTER
# =====================================================

start_date = st.sidebar.date_input(
    "📅 Start Date",
    value=df["Date"].min()
)

end_date = st.sidebar.date_input(
    "📅 End Date",
    value=df["Date"].max()
)

# =====================================================
# APPLY FILTERS
# =====================================================

filtered_df = df[
    (df["Category"].isin(category)) &
    (df["Transaction Type"].isin(transaction)) &
    (df["Account Name"].isin(account))
]

filtered_df = filtered_df[
    (filtered_df["Date"] >= pd.to_datetime(start_date)) &
    (filtered_df["Date"] <= pd.to_datetime(end_date))
]

# =====================================================
# SIDEBAR STATISTICS
# =====================================================

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Quick Statistics")

st.sidebar.metric(
    "Transactions",
    len(filtered_df)
)

st.sidebar.metric(
    "Categories",
    filtered_df["Category"].nunique()
)

st.sidebar.metric(
    "Accounts",
    filtered_df["Account Name"].nunique()
)

st.sidebar.metric(
    "Average Amount",
    f"₹ {filtered_df['Amount'].mean():,.2f}"
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
### 📌 About

**Personal Finance Transaction Dashboard**

Built using:

✅ Python

✅ Streamlit

✅ Plotly

✅ Machine Learning

✅ Pandas
"""
)

# =====================================================
# PAGE TITLE
# =====================================================

st.title("💰 Personal Finance Transaction Dashboard")

st.write(
"""
Analyze your personal transactions using interactive
visualizations and Machine Learning.

Filter transactions, explore spending trends,
and predict transaction types.
"""
)

st.markdown("---")

# =====================================================
# CREATE TABS
# =====================================================

dashboard, analytics, prediction = st.tabs(

    [
        "📊 Dashboard",
        "📈 Analytics",
        "🤖 Prediction"
    ]

)

# =====================================================
# DASHBOARD TAB
# =====================================================

with dashboard:

    st.header("📊 Dashboard")

    # -------------------------------
    # KPI CARDS
    # -------------------------------

    income = filtered_df[
        filtered_df["Transaction Type"] == "Income"
    ]["Amount"].sum()

    expense = filtered_df[
        filtered_df["Transaction Type"] == "Expense"
    ]["Amount"].sum()

    balance = income - expense

    transactions = len(filtered_df)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "💵 Total Income",
        f"₹ {income:,.2f}"
    )

    c2.metric(
        "💸 Total Expense",
        f"₹ {expense:,.2f}"
    )

    c3.metric(
        "💰 Balance",
        f"₹ {balance:,.2f}"
    )

    c4.metric(
        "🧾 Transactions",
        transactions
    )

    st.markdown("---")

    # -------------------------------
    # PIE CHART
    # -------------------------------

    left, right = st.columns(2)

    with left:

        pie = filtered_df.groupby(
            "Category"
        )["Amount"].sum().reset_index()

        fig = px.pie(
            pie,
            names="Category",
            values="Amount",
            hole=0.5,
            title="Expense by Category"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -------------------------------
    # BAR CHART
    # -------------------------------

    with right:

        bar = filtered_df.groupby(
            "Account Name"
        )["Amount"].sum().reset_index()

        fig = px.bar(
            bar,
            x="Account Name",
            y="Amount",
            color="Account Name",
            title="Account Wise Spending"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # -------------------------------
    # MONTHLY TREND
    # -------------------------------

    st.subheader("📈 Monthly Spending Trend")

    monthly = filtered_df.groupby(
        filtered_df["Date"].dt.to_period("M")
    )["Amount"].sum().reset_index()

    monthly["Date"] = monthly["Date"].astype(str)

    fig = px.line(
        monthly,
        x="Date",
        y="Amount",
        markers=True,
        title="Monthly Expense Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # -------------------------------
    # HISTOGRAM & BOXPLOT
    # -------------------------------

    left, right = st.columns(2)

    with left:

        fig = px.histogram(
            filtered_df,
            x="Amount",
            nbins=30,
            color="Transaction Type",
            title="Amount Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        fig = px.box(
            filtered_df,
            x="Transaction Type",
            y="Amount",
            color="Transaction Type",
            title="Transaction Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # -------------------------------
    # CATEGORY VS TRANSACTION
    # -------------------------------

    st.subheader("📊 Category vs Transaction Type")

    fig = px.histogram(
        filtered_df,
        x="Category",
        color="Transaction Type",
        barmode="group",
        title="Category-wise Transactions"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # -------------------------------
    # SEARCH
    # -------------------------------

    st.subheader("🔍 Search Transactions")

    search = st.text_input(
        "Search Description"
    )

    table_df = filtered_df.copy()

    if search:

        table_df = table_df[
            table_df["Description"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    # -------------------------------
    # DATA TABLE
    # -------------------------------

    st.subheader("📋 Transaction Details")

    st.dataframe(
        table_df,
        use_container_width=True,
        height=450
    )

    # -------------------------------
    # DOWNLOAD BUTTON
    # -------------------------------

    csv = table_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇ Download Filtered CSV",
        data=csv,
        file_name="filtered_transactions.csv",
        mime="text/csv"
    )
    
# =====================================================
# ANALYTICS TAB
# =====================================================

with analytics:

    st.header("📈 Advanced Analytics")

    st.write(
        """
        Explore your financial transactions using advanced
        interactive visualizations.
        """
    )

    # --------------------------------------------
    # TREEMAP & SUNBURST
    # --------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.treemap(
            filtered_df,
            path=["Category"],
            values="Amount",
            color="Amount",
            color_continuous_scale="Blues",
            title="Treemap of Expenses"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.sunburst(
            filtered_df,
            path=["Transaction Type", "Category"],
            values="Amount",
            color="Amount",
            title="Transaction Hierarchy"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    # --------------------------------------------
    # CATEGORY SUMMARY
    # --------------------------------------------

    st.subheader("📋 Category Summary")

    category_summary = filtered_df.groupby(
        "Category"
    )["Amount"].agg(
        Total="sum",
        Average="mean",
        Maximum="max",
        Minimum="min",
        Transactions="count"
    ).reset_index()

    st.dataframe(
        category_summary,
        use_container_width=True
    )

    st.markdown("---")

    # --------------------------------------------
    # ACCOUNT SUMMARY
    # --------------------------------------------

    st.subheader("🏦 Account Summary")

    account_summary = filtered_df.groupby(
        "Account Name"
    )["Amount"].agg(
        Total="sum",
        Average="mean",
        Transactions="count"
    ).reset_index()

    st.dataframe(
        account_summary,
        use_container_width=True
    )

    st.markdown("---")

    # --------------------------------------------
    # MONTHLY SUMMARY
    # --------------------------------------------

    st.subheader("📅 Monthly Summary")

    monthly_summary = filtered_df.copy()

    monthly_summary["Month"] = (
        monthly_summary["Date"]
        .dt.to_period("M")
        .astype(str)
    )

    month_table = monthly_summary.groupby(
        "Month"
    )["Amount"].agg(
        Total="sum",
        Average="mean",
        Transactions="count"
    ).reset_index()

    st.dataframe(
        month_table,
        use_container_width=True
    )

    st.markdown("---")

    # --------------------------------------------
    # CATEGORY BAR CHART
    # --------------------------------------------

    st.subheader("💵 Category Wise Spending")

    cat = filtered_df.groupby(
        "Category"
    )["Amount"].sum().reset_index()

    fig = px.bar(
        cat,
        x="Category",
        y="Amount",
        color="Category",
        text_auto=True,
        title="Category Wise Amount"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # --------------------------------------------
    # ACCOUNT PIE CHART
    # --------------------------------------------

    st.subheader("🏦 Account Contribution")

    acc = filtered_df.groupby(
        "Account Name"
    )["Amount"].sum().reset_index()

    fig = px.pie(
        acc,
        names="Account Name",
        values="Amount",
        hole=0.45,
        title="Account Contribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # --------------------------------------------
    # HEATMAP (PIVOT TABLE)
    # --------------------------------------------

    st.subheader("🔥 Category vs Transaction Type")

    pivot = pd.pivot_table(
        filtered_df,
        values="Amount",
        index="Category",
        columns="Transaction Type",
        aggfunc="sum",
        fill_value=0
    )

    fig = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        title="Transaction Heatmap"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # --------------------------------------------
    # TOP 10 TRANSACTIONS
    # --------------------------------------------

    st.subheader("🏆 Top 10 Highest Transactions")

    top10 = filtered_df.sort_values(
        by="Amount",
        ascending=False
    ).head(10)

    st.dataframe(
        top10,
        use_container_width=True
    )

    st.markdown("---")

    # --------------------------------------------
    # INSIGHTS
    # --------------------------------------------

    st.subheader("📌 Financial Insights")

    highest_category = (
        filtered_df.groupby("Category")["Amount"]
        .sum()
        .idxmax()
    )

    highest_account = (
        filtered_df.groupby("Account Name")["Amount"]
        .sum()
        .idxmax()
    )

    st.success(
        f"💡 Highest spending category: **{highest_category}**"
    )

    st.info(
        f"🏦 Most used account: **{highest_account}**"
    )

    st.warning(
        f"💰 Total Balance: ₹ {balance:,.2f}"
    )
    
# =====================================================
# PREDICTION TAB
# =====================================================

with prediction:

    st.header("🤖 Transaction Type Prediction")

    st.write(
        """
        Predict whether a transaction is **Income** or **Expense**
        using the trained Machine Learning model.
        """
    )

    st.markdown("---")

    # --------------------------------------------
    # INPUTS
    # --------------------------------------------

    amount = st.number_input(
        "💰 Transaction Amount",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    category = st.selectbox(
        "📂 Category",
        sorted(df["Category"].unique())
    )

    account = st.selectbox(
        "🏦 Account Name",
        sorted(df["Account Name"].unique())
    )

    st.markdown("---")

    # --------------------------------------------
    # ENCODE INPUTS
    # --------------------------------------------

    category_encoded = category_encoder.transform([category])[0]
    account_encoded = account_encoder.transform([account])[0]
    prediction = model.predict([[amount, category_encoded, account_encoded]])
    
    # --------------------------------------------
    # PREDICT
    # --------------------------------------------

    if st.button("🔮 Predict Transaction Type"):

        input_data = [[
            amount,
            category_encoded,
            account_encoded
        ]]

        prediction_result = model.predict(input_data)[0]

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction_result == 0:

            st.error("💸 Predicted Transaction Type: Expense")

        else:

            st.success("💵 Predicted Transaction Type: Income")

        # Probability (only if supported)

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(input_data)

            st.write("Prediction Confidence")

            st.progress(
                float(max(probability[0]))
            )

            st.write(
                f"Confidence : {max(probability[0])*100:.2f}%"
            )

        st.markdown("---")

        st.subheader("Transaction Preview")

        preview = pd.DataFrame({

            "Amount":[amount],

            "Category":[category],

            "Account":[account]

        })

        st.dataframe(
            preview,
            use_container_width=True
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
"""
<div style="text-align:center">

### 💰 Personal Finance Transaction Dashboard

Built using

**Python • Streamlit • Plotly • Scikit-Learn • XGBoost**

---

Created by **Saheranjum Makandar**

</div>
""",
unsafe_allow_html=True
)