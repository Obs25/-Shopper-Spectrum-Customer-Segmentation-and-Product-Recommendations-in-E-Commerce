import streamlit as st
st.set_page_config(page_title="🛒 Shopper Spectrum", layout="wide")

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data & Models
@st.cache_data
def load_data():
    df = pd.read_csv("online_retail.csv", encoding="ISO-8859-1")
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df

@st.cache_resource
def load_models():
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    item_sim_df = pd.read_pickle("item_similarity.pkl")
    product_map = pd.read_pickle("product_map.pkl")
    return kmeans, scaler, item_sim_df, product_map

df = load_data()
kmeans, scaler, item_sim_df, product_map = load_models()
product_map_cleaned = product_map.apply(lambda x: x.strip())
product_reverse_map = product_map_cleaned.reset_index().drop_duplicates(subset='Description').set_index('Description')['StockCode']

# Sidebar Nav
st.markdown("<h1 style='text-align: center; color: #0F9D58;'>Shopper Spectrum</h1>", unsafe_allow_html=True)
st.markdown("### 💼 Customer Segmentation & 🛍️ Product Recommendations")
st.sidebar.title("🔎 Navigation")
module = st.sidebar.radio("Choose a module:", ["📊 Dashboard", "🎯 Product Recommendation", "🔍 Customer Segmentation"])

# Dashboard
if module == "📊 Dashboard":
    st.subheader("📊 Interactive Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        top_countries = df["Country"].value_counts().head(10)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax1, palette="cubehelix")
        ax1.set_title("Top Countries by Transactions")
        st.pyplot(fig1)

    with col2:
        top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=top_products.values, y=top_products.index, ax=ax2, palette="rocket")
        ax2.set_title("Top 10 Best-Selling Products")
        st.pyplot(fig2)

    monthly = df.copy()
    monthly["MonthYear"] = monthly["InvoiceDate"].dt.to_period("M")
    sales_trend = monthly.groupby("MonthYear")["InvoiceNo"].nunique()

    st.markdown("### 🗓️ Monthly Sales Trend")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sales_trend.plot(marker="o", ax=ax3, color='mediumblue')
    ax3.set_title("Monthly Transactions")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Transactions")
    ax3.grid(True)
    st.pyplot(fig3)

# Product Recommendation
elif module == "🎯 Product Recommendation":
    st.subheader("🎯 Find Similar Products")

    product_input = st.selectbox("Select a Product Name:", options=sorted(product_reverse_map.index.tolist()))
    product_code = product_reverse_map.get(product_input)

    if st.button("Recommend Similar Products"):
        if product_code in item_sim_df.columns:
            similar_items = item_sim_df[product_code].sort_values(ascending=False)[1:6]
            st.success("🔁 Top 5 Similar Products:")
            for idx in similar_items.index:
                name = product_map.get(idx, f"StockCode: {idx}").strip()
                st.markdown(f"✅ **{name}**")
        else:
            st.warning("Product not found in similarity matrix.")

# Customer Segmentation
elif module == "🔍 Customer Segmentation":
    st.subheader("🔍 Predict Customer Segment")

    recency = st.slider("Recency (days since last purchase):", 0, 500, 30)
    frequency = st.slider("Frequency (number of purchases):", 1, 100, 10)
    monetary = st.slider("Monetary (total spend):", 1, 10000, 500)

    if st.button("Predict Segment"):
        input_scaled = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(input_scaled)[0]
        cluster_labels = {
            0: 'Regular',
            1: 'At-Risk',
            2: 'High-Value',
            3: 'Occasional'
        }
        segment = cluster_labels.get(cluster, "Unknown")
        st.success(f"🧠 Predicted Segment: **{segment}**")
