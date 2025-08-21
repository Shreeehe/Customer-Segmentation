# 🛍️ Customer Segmentation Using Unsupervised Learning

This project applies **Unsupervised Machine Learning** (Clustering) to
segment customers based on **Age, Gender, Annual Income, and Spending
Score**.\
It helps businesses design **personalized marketing campaigns, loyalty
programs, and product recommendations**.

------------------------------------------------------------------------

## 📊 Project Overview

-   **Goal:** Segment customers into distinct groups based on purchasing
    behavior.\
-   **Domain:** Retail, E-Commerce, Marketing Analytics.\
-   **Dataset:** Mall Customers Dataset (200 records).

------------------------------------------------------------------------

## 🚀 Features

✔️ Data Cleaning & Preprocessing\
✔️ Exploratory Data Analysis (EDA) with interactive charts\
✔️ Clustering (KMeans, DBSCAN, Hierarchical)\
✔️ Evaluation Metrics (Silhouette, Davies-Bouldin, Inertia)\
✔️ Dimensionality Reduction (PCA 2D/3D, t-SNE)\
✔️ Streamlit Dashboard for interactive exploration\
✔️ Business Insights & Recommendations

------------------------------------------------------------------------

## 📂 Project Structure

    ├── app.py                     # Streamlit Dashboard
    ├── Final Project 2 Mall Customer Dataset.csv   # Dataset
    ├── notebooks/                 # Jupyter Notebooks (EDA, Clustering, Evaluation)
    ├── results/                   # Saved plots & cluster summaries
    └── README.md                  # Documentation

------------------------------------------------------------------------

## 🛠️ Installation & Setup

Clone the repo:

``` bash
git clone https://github.com/your-username/customer-segmentation.git
cd customer-segmentation
```

Create environment & install dependencies:

``` bash
pip install -r requirements.txt
```

Run the Streamlit Dashboard:

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 📊 Dashboard Preview

The **Streamlit Dashboard** includes: - **EDA Tab** → Histograms,
Heatmap\
- **Clustering Tab** → KMeans, DBSCAN, Hierarchical + Evaluation
metrics\
- **Visualization Tab** → PCA (2D/3D), t-SNE\
- **Interactive Analytics Tab** → Filter by Age, Income, Spending\
- **Insights Tab** → Business segmentation strategy

------------------------------------------------------------------------

## 📈 Results

-   **Optimal K (clusters):** 5\
-   **Silhouette Score:** \~0.27\
-   **Davies-Bouldin Index:** \~1.18\
-   **Key Insights:**
    -   High-income, low-spending customers need re-engagement.\
    -   Young high-spenders are premium targets.\
    -   Middle-aged moderate spenders respond to family/comfort deals.

------------------------------------------------------------------------

## 🧩 Deliverables

1.  ✅ Data Cleaning Script\
2.  ✅ EDA Notebook (EDA + visualizations)\
3.  ✅ Clustering Code (KMeans, DBSCAN, Hierarchical)\
4.  ✅ Evaluation Report (Silhouette, DB Index, Inertia)\
5.  ✅ Visualization Script (PCA/t-SNE plots)\
6.  ✅ Segmentation Strategy & Insights\
7.  ✅ Interactive Streamlit Dashboard\
8.  ✅ Documentation (README.md)

------------------------------------------------------------------------

## 📌 Business Impact

This project empowers businesses to: - 🎯 Run **targeted marketing**
campaigns\
- 🛍️ Improve **product recommendations**\
- 💳 Increase **customer retention**\
- 📦 Optimize **inventory planning**\
- 🎁 Design **loyalty programs**

------------------------------------------------------------------------

## 📜 License

This project is licensed under the **MIT License** -- feel free to use
and modify it.

------------------------------------------------------------------------

👩‍💻 **Author:** Kitty\
✨ **Technologies:** Python, Scikit-Learn, Pandas, Matplotlib, Seaborn,
Plotly, Streamlit
