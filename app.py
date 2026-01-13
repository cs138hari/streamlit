# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Data Mining & ML App",
    layout="wide"
)

st.title("📊 Data Mining & Machine Learning App")

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

df = None
if uploaded_file is not None:
    df = load_csv(uploaded_file)

# =====================================================
# REQUIRED COLUMNS CHECK
# =====================================================
required_cols = [
    "InvoiceID",
    "InvoiceDate",
    "CustomerID",
    "ItemName",
    "Quantity",
    "UnitPrice",
    "TotalAmount"
]

if df is not None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.stop()

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📌 RFM Analysis",
    "📊 K-Means Clustering",
    "🔗 Apriori Algorithm",
    "🔗 Hybrid K-Means + Apriori"
])

# =====================================================
# TAB 1: RFM ANALYSIS
# =====================================================
with tab1:
    if df is not None:
        st.header("📌 RFM Analysis")

        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

        rfm = (
            df.groupby("CustomerID")
            .agg({
                "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
                "InvoiceID": "count",
                "TotalAmount": "sum"
            })
            .reset_index()
        )

        rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
        st.dataframe(rfm.head())

        # Scaling
        scaler = StandardScaler()
        rfm[["R_Scaled", "F_Scaled", "M_Scaled"]] = scaler.fit_transform(
            rfm[["Recency", "Frequency", "Monetary"]]
        )

        # RFM Scores
        rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1])
        rfm["F_Score"] = pd.qcut(rfm["Frequency"], 5, labels=[1,2,3,4,5])
        rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5])
        rfm["RFM_Total"] = (
            rfm["R_Score"].astype(int) +
            rfm["F_Score"].astype(int) +
            rfm["M_Score"].astype(int)
        )

        # Metrics
        st.subheader("📈 RFM Evaluation Metrics")
        st.dataframe(rfm[["Recency","Frequency","Monetary"]].describe())

        cv = rfm[["Recency","Frequency","Monetary"]].std() / rfm[["Recency","Frequency","Monetary"]].mean()
        st.dataframe(cv.to_frame("Coefficient of Variation"))

        st.dataframe(rfm[["Recency","Frequency","Monetary"]].skew().to_frame("Skewness"))
        st.dataframe(rfm[["Recency","Frequency","Monetary"]].kurtosis().to_frame("Kurtosis"))

        pareto = (
            rfm.sort_values("Monetary", ascending=False)
            .iloc[:int(0.2 * len(rfm))]["Monetary"].sum()
            / rfm["Monetary"].sum()
        )
        st.metric("Revenue from Top 20% Customers", f"{pareto:.2%}")

        # Visuals
        fig, ax = plt.subplots(1,3,figsize=(15,4))
        sns.histplot(rfm["Recency"], ax=ax[0])
        sns.histplot(rfm["Frequency"], ax=ax[1])
        sns.histplot(rfm["Monetary"], ax=ax[2])
        st.pyplot(fig)

# =====================================================
# TAB 2: K-MEANS
# =====================================================
with tab2:
    if df is not None:
        st.header("📊 K-Means Clustering")

        num_cols = ["Quantity", "UnitPrice", "TotalAmount"]
        data = df[num_cols]

        scaled = StandardScaler().fit_transform(data)

        # Elbow
        wcss = []
        for k in range(2,11):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(scaled)
            wcss.append(km.inertia_)

        fig, ax = plt.subplots()
        ax.plot(range(2,11), wcss, marker="o")
        st.pyplot(fig)

        k = st.slider("Select K", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled)
        df["Cluster"] = clusters

        st.metric("Silhouette Score", f"{silhouette_score(scaled, clusters):.4f}")
        st.metric("Davies–Bouldin", f"{davies_bouldin_score(scaled, clusters):.4f}")

        pca = PCA(2)
        pca_data = pca.fit_transform(scaled)
        fig, ax = plt.subplots()
        ax.scatter(pca_data[:,0], pca_data[:,1], c=clusters)
        st.pyplot(fig)

# =====================================================
# TAB 3: APRIORI
# =====================================================
with tab3:
    if df is not None:
        st.header("🔗 Apriori Algorithm")

        basket = (
            df.groupby(["InvoiceID","ItemName"])["Quantity"]
            .sum().unstack().fillna(0)
        )

        basket = (basket > 0).astype(int)
        freq = apriori(basket, min_support=0.005, use_colnames=True)

        if freq.empty:
            st.warning("No frequent itemsets found")
        else:
            rules = association_rules(freq, metric="confidence", min_threshold=0.3)
            st.dataframe(rules.head())

# =====================================================
# TAB 4: HYBRID MODEL
# =====================================================
with tab4:
    if df is not None:
        st.header("🔗 Hybrid K-Means + Apriori")

        scaled = StandardScaler().fit_transform(df[["Quantity","TotalAmount"]])
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["HybridCluster"] = kmeans.fit_predict(scaled)

        cluster = st.selectbox("Select Cluster", sorted(df["HybridCluster"].unique()))
        cluster_df = df[df["HybridCluster"] == cluster]

        basket = (
            cluster_df.groupby(["InvoiceID","ItemName"])["Quantity"]
            .sum().unstack().fillna(0)
        )

        basket = (basket > 0).astype(int)

        freq = apriori(basket, min_support=0.01, use_colnames=True)
        if freq.empty:
            st.warning("No rules found")
        else:
            rules = association_rules(freq, metric="confidence", min_threshold=0.1)
            st.dataframe(rules.sort_values("lift", ascending=False).head(10))
