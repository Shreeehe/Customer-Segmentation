# ----------------------------------------
# Advanced Customer Segmentation Dashboard
# ----------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram

# Streamlit Extras
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

# ----------------------------------------
# 1. Load Data
# ----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Final Project 2 Mall Customer Dataset.csv")
    return df

df = load_data()

# Sidebar
st.sidebar.title("⚙️ Settings")
show_raw = st.sidebar.checkbox("Show Raw Data")
if show_raw:
    st.write("### Raw Dataset")
    st.dataframe(df)

# Clean & scale
df_clean = df.drop("CustomerID", axis=1)
le = LabelEncoder()
df_clean["Gender"] = le.fit_transform(df_clean["Gender"])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean)

# ----------------------------------------
# Tabs
# ----------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 EDA", "🤖 Clustering", "📈 Visualization", "📊 Interactive Analytics", "💡 Insights"])

# ----------------------------------------
# TAB 1: EDA
# ----------------------------------------
with tab1:
    st.header("📊 Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.histogram(df, x="Age", nbins=20, title="Age Distribution", color="Gender")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="Annual Income (k$)", nbins=20, title="Income Distribution", color="Gender")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = px.histogram(df, x="Spending Score (1-100)", nbins=20, title="Spending Score Distribution", color="Gender")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(df_clean.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ----------------------------------------
# TAB 2: Clustering
# ----------------------------------------
with tab2:
    st.header("🤖 Clustering Models")
    algo = st.radio("Choose Algorithm", ["KMeans", "DBSCAN", "Hierarchical"])

    if algo == "KMeans":
        K = st.slider("Select Clusters (K)", 2, 10, 5)
        kmeans = KMeans(n_clusters=K, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        df_clean["Cluster"] = clusters

        # Styled metric cards
        # Styled metric cards for dark mode
        col1, col2, col3 = st.columns(3)
        col1.metric("Silhouette Score", f"{silhouette_score(scaled_data, clusters):.2f}")
        col2.metric("Davies-Bouldin Index", f"{davies_bouldin_score(scaled_data, clusters):.2f}")
        col3.metric("Inertia", f"{kmeans.inertia_:.2f}")

        style_metric_cards(
            background_color="#1e1e1e",   # dark grey card
            border_color="#444444",       # subtle border
            border_left_color="#00bfff",  # bright cyan accent
            box_shadow=True
        )


        st.subheader("Cluster Summary")
        st.dataframe(df_clean.groupby("Cluster").mean())

    elif algo == "DBSCAN":
        eps = st.slider("Epsilon", 0.5, 5.0, 1.5)
        min_samples = st.slider("Min Samples", 2, 15, 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_data)
        df_clean["Cluster"] = clusters
        st.write("DBSCAN Clusters:", set(clusters))

    elif algo == "Hierarchical":
        st.subheader("Hierarchical Clustering Dendrogram")
        linked = linkage(scaled_data, method='ward')
        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(linked, ax=ax)
        st.pyplot(fig)

# ----------------------------------------
# TAB 3: Visualization
# ----------------------------------------
with tab3:
    st.header("📈 Cluster Visualization")

    if "Cluster" not in df_clean.columns:
        st.warning("⚠️ Run clustering first in Clustering tab.")
    else:
        method = st.radio("Choose Method", ["PCA 2D", "PCA 3D", "t-SNE"])

        if method == "PCA 2D":
            pca = PCA(2)
            pca_data = pca.fit_transform(scaled_data)
            fig = px.scatter(x=pca_data[:,0], y=pca_data[:,1], color=df_clean["Cluster"].astype(str),
                             title="PCA 2D Clusters", labels={"x":"PCA1","y":"PCA2"})
            st.plotly_chart(fig, use_container_width=True)

        elif method == "PCA 3D":
            pca = PCA(3)
            pca_data = pca.fit_transform(scaled_data)
            fig = px.scatter_3d(x=pca_data[:,0], y=pca_data[:,1], z=pca_data[:,2],
                                color=df_clean["Cluster"].astype(str), title="PCA 3D Clusters")
            st.plotly_chart(fig, use_container_width=True)

        elif method == "t-SNE":
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            tsne_data = tsne.fit_transform(scaled_data)
            fig = px.scatter(x=tsne_data[:,0], y=tsne_data[:,1], color=df_clean["Cluster"].astype(str),
                             title="t-SNE Clusters")
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------
# TAB 4: Interactive Analytics
# ----------------------------------------
with tab4:
    st.header("📊 Interactive Analytics")

    st.write("Filter customers by Age, Income, and Spending Score:")

    age_filter = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), (20,40))
    income_filter = st.slider("Annual Income (k$)", int(df["Annual Income (k$)"].min()), int(df["Annual Income (k$)"].max()), (20,80))
    spend_filter = st.slider("Spending Score", int(df["Spending Score (1-100)"].min()), int(df["Spending Score (1-100)"].max()), (20,80))

    df_filtered = df[(df["Age"].between(age_filter[0], age_filter[1])) &
                     (df["Annual Income (k$)"].between(income_filter[0], income_filter[1])) &
                     (df["Spending Score (1-100)"].between(spend_filter[0], spend_filter[1]))]

    st.write(f"Showing {df_filtered.shape[0]} customers after filter:")
    st.dataframe(df_filtered)

    fig = px.scatter(df_filtered, x="Annual Income (k$)", y="Spending Score (1-100)", color="Gender", size="Age",
                     title="Filtered Customers: Income vs Spending")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------
# TAB 5: Insights
# ----------------------------------------
with tab5:
    st.header("💡 Business Segmentation Strategy")

    st.markdown("""
    - **Cluster 0** → Middle-aged, moderate spenders → Target with comfort/family deals.  
    - **Cluster 1** → High income, low spenders → Offer special discounts to increase spending.  
    - **Cluster 2** → Young high spenders → Premium offers, loyalty rewards.  
    - **Cluster 3** → High income mid spenders → Upsell luxury products.  
    - **Cluster 4** → Young, trend-driven → Flash sales & new collections.  
    """)
