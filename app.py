import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Data Mining ML App", layout="wide")
st.title("📊 Data Mining & Machine Learning App")

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

df = None
if uploaded_file is not None:
    df = load_csv(uploaded_file)

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

        scaler = StandardScaler()
        rfm[["R_Scaled", "F_Scaled", "M_Scaled"]] = scaler.fit_transform(
            rfm[["Recency", "Frequency", "Monetary"]]
        )

        rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
        rfm["F_Score"] = pd.qcut(rfm["Frequency"], 5, labels=[1, 2, 3, 4, 5])
        rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

        rfm["RFM_Total"] = (
            rfm["R_Score"].astype(int)
            + rfm["F_Score"].astype(int)
            + rfm["M_Score"].astype(int)
        )

        st.subheader("📈 RFM Evaluation Metrics")
        st.dataframe(rfm[["Recency", "Frequency", "Monetary"]].describe())

        cv = rfm[["Recency", "Frequency", "Monetary"]].std() / \
             rfm[["Recency", "Frequency", "Monetary"]].mean()
        st.dataframe(cv.to_frame("Coefficient of Variation"))

        st.subheader("📊 RFM Distributions")
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(rfm["Recency"], bins=30, ax=ax[0])
        sns.histplot(rfm["Frequency"], bins=30, ax=ax[1])
        sns.histplot(rfm["Monetary"], bins=30, ax=ax[2])
        st.pyplot(fig)

    else:
        st.info("Upload CSV file")

# =====================================================
# TAB 2: K-MEANS CLUSTERING (FIXED)
# =====================================================
with tab2:
    if df is not None:
        st.header("📊 K-Means Clustering")

        st.dataframe(df.head())

        num_cols = ["Quantity", "UnitPrice", "TotalAmount"]
        data = df[num_cols]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        wcss = []
        K_RANGE = range(0, 11)

        for k_val in K_RANGE:
            if k_val < 2:
                wcss.append(np.nan)
            else:
                km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
                km.fit(scaled_data)
                wcss.append(km.inertia_)

        fig, ax = plt.subplots()
        ax.plot(K_RANGE, wcss, marker="o")
        ax.set_xlabel("K")
        ax.set_ylabel("WCSS")
        st.pyplot(fig)

        k = st.slider("Select number of clusters (K)", 0, 10, 3)

        if k < 2:
            st.warning("⚠️ K-Means requires K ≥ 2")
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
            df["KMeans_Cluster"] = clusters

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("WCSS", f"{kmeans.inertia_:.2f}")
            col2.metric("Silhouette", f"{silhouette_score(scaled_data, clusters):.4f}")
            col3.metric("DB Index", f"{davies_bouldin_score(scaled_data, clusters):.4f}")
            col4.metric("CH Index", f"{calinski_harabasz_score(scaled_data, clusters):.2f}")

            fig2, ax2 = plt.subplots()
            ax2.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters)
            st.pyplot(fig2)

    else:
        st.info("Upload CSV file")

# =====================================================
# TAB 3: APRIORI
# =====================================================
with tab3:
    if df is not None:
        basket = (
            df.groupby(["InvoiceID", "ItemName"])["Quantity"]
            .sum().unstack().fillna(0)
        )
        basket = (basket > 0).astype(int)

        freq = apriori(basket, min_support=0.005, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=0.3)

        st.dataframe(rules.head())
    else:
        st.info("Upload CSV file")

# =====================================================
# TAB 4: HYBRID
# =====================================================
with tab4:
    if df is not None:
        num_scaled = StandardScaler().fit_transform(df[["Quantity", "TotalAmount"]])
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["Cluster"] = kmeans.fit_predict(num_scaled)

        cluster = st.selectbox("Select Cluster", sorted(df["Cluster"].unique()))
        cluster_df = df[df["Cluster"] == cluster]

        basket = (
            cluster_df.groupby(["InvoiceID", "ItemName"])["Quantity"]
            .sum().unstack().fillna(0)
        )
        basket = (basket > 0).astype(int)

        freq = apriori(basket, min_support=0.003, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=0.1)

        st.dataframe(rules.head())
    else:
        st.info("Upload CSV file")
