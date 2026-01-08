import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

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

        rfm = df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
            "InvoiceID": "count",
            "TotalAmount": "sum"
        }).reset_index()

        rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
        st.dataframe(rfm.head())

        scaler = StandardScaler()
        rfm[["R_Scaled","F_Scaled","M_Scaled"]] = scaler.fit_transform(
            rfm[["Recency","Frequency","Monetary"]]
        )

        rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1])
        rfm["F_Score"] = pd.qcut(rfm["Frequency"], 5, labels=[1,2,3,4,5])
        rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5])
        rfm["RFM_Total"] = (
            rfm["R_Score"].astype(int)
            + rfm["F_Score"].astype(int)
            + rfm["M_Score"].astype(int)
        )

        st.subheader("📊 RFM Distributions")
        fig_rfm, ax = plt.subplots(1, 3, figsize=(15,4))
        sns.histplot(rfm["Recency"], bins=30, ax=ax[0])
        sns.histplot(rfm["Frequency"], bins=30, ax=ax[1])
        sns.histplot(rfm["Monetary"], bins=30, ax=ax[2])
        st.pyplot(fig_rfm)

        st.subheader("📊 RFM Score Distribution")
        fig_score, ax_score = plt.subplots()
        rfm["RFM_Total"].value_counts().sort_index().plot(kind="bar", ax=ax_score)
        st.pyplot(fig_score)

    else:
        st.info("Upload CSV file")

# =====================================================
# TAB 2: K-MEANS CLUSTERING
# =====================================================
with tab2:
    if df is not None:
        st.header("📊 K-Means Clustering")

        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        # ---------- Select Numeric Columns ----------
        num_cols = ["Quantity", "UnitPrice", "TotalAmount"]
        data = df[num_cols]

        st.write("Numeric columns used:", num_cols)

        # ---------- Scaling ----------
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # ---------- Elbow Method ----------
        wcss = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            wcss.append(kmeans.inertia_)

        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(range(2, 11), wcss, marker="o")
        ax_elbow.set_xlabel("Number of Clusters (K)")
        ax_elbow.set_ylabel("WCSS")
        ax_elbow.set_title("Elbow Method")
        st.pyplot(fig_elbow)

        # ---------- Choose K ----------
        k = st.slider("Select number of clusters (K)", 2, 10, 3)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)

        df["KMeans_Cluster"] = clusters

        # ---------- Evaluation ----------
        score = silhouette_score(scaled_data, clusters)
        st.success(f"Silhouette Score: {score:.3f}")

        # ---------- PCA for Visualization ----------
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)

        fig_scatter, ax_scatter = plt.subplots()
        scatter = ax_scatter.scatter(
            pca_data[:, 0],
            pca_data[:, 1],
            c=clusters,
            cmap="viridis"
        )
        ax_scatter.set_xlabel("PCA 1")
        ax_scatter.set_ylabel("PCA 2")
        ax_scatter.set_title("K-Means Clusters (PCA View)")
        st.pyplot(fig_scatter)

        # ---------- Cluster Distribution ----------
        st.subheader("📊 Cluster Size Distribution")
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        st.bar_chart(cluster_counts)

    else:
        st.info("Upload CSV file")

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
        rules = association_rules(freq, metric="confidence", min_threshold=0.3)

        if not rules.empty:
            st.dataframe(rules.head())

            st.subheader("📊 Apriori Visualizations")
            fig1, ax1 = plt.subplots()
            ax1.scatter(rules["support"], rules["confidence"])
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            ax2.hist(rules["lift"], bins=20)
            st.pyplot(fig2)
        else:
            st.warning("No rules found")

    else:
        st.info("Upload CSV file")

# =====================================================
# TAB 4: HYBRID K-MEANS + APRIORI
# =====================================================
with tab4:
    if df is not None:
        st.header("🔗 Hybrid K-Means + Apriori (Cluster-wise Rules)")

        # ---------- K-Means ----------
        scaled = StandardScaler().fit_transform(df[["Quantity", "TotalAmount"]])
        df["Cluster"] = KMeans(n_clusters=3, random_state=42).fit_predict(scaled)

        # ---------- Cluster Distribution ----------
        st.subheader("📊 Cluster Distribution")
        cluster_counts = df["Cluster"].value_counts().sort_index()

        fig_bar, ax_bar = plt.subplots()
        cluster_counts.plot(kind="bar", ax=ax_bar)
        st.pyplot(fig_bar)

        # ---------- Select Cluster ----------
        selected_cluster = st.selectbox(
            "Select Cluster for Apriori",
            sorted(df["Cluster"].unique())
        )

        st.markdown(f"### 📌 Association Rules for **Cluster {selected_cluster}**")

        # ---------- Filter Cluster ----------
        cluster_df = df[df["Cluster"] == selected_cluster]

        st.write("### Cluster Statistics")
        st.write("Total Invoices:", cluster_df["InvoiceID"].nunique())
        st.write("Unique Items:", cluster_df["ItemName"].nunique())

        # ---------- Remove single-item invoices ----------
        invoice_item_count = cluster_df.groupby("InvoiceID")["ItemName"].nunique()
        valid_invoices = invoice_item_count[invoice_item_count >= 2].index
        cluster_df = cluster_df[cluster_df["InvoiceID"].isin(valid_invoices)]

        # ---------- Basket ----------
        basket = (
            cluster_df
            .groupby(["InvoiceID", "ItemName"])["Quantity"]
            .sum()
            .unstack()
            .fillna(0)
        )

        basket = (basket > 0).astype(int)
        st.write("Basket Shape:", basket.shape)

        # ---------- Apriori ----------
        min_support = max(1 / basket.shape[0], 0.003)

        freq_items = apriori(
            basket,
            min_support=min_support,
            use_colnames=True
        )

        st.write("Total frequent itemsets:", len(freq_items))

        if freq_items.empty:
            st.warning("❌ No frequent itemsets found")
        else:
            rules = association_rules(
                freq_items,
                metric="confidence",
                min_threshold=0.1
            )

            rules = rules[
                (rules["antecedents"].apply(len) >= 1) &
                (rules["consequents"].apply(len) >= 1)
            ]

            st.write("Association rules formed:", len(rules))

            if rules.empty:
                st.warning("❌ No valid association rules formed")
            else:
                st.subheader("### ✅ Association Rules")
                st.dataframe(
                    rules[
                        ["antecedents", "consequents", "support", "confidence", "lift"]
                    ]
                    .sort_values("lift", ascending=False)
                    .head(10)
                )

                # ---------- Visualizations ----------
                st.subheader("📊 Rule Visualizations")

                fig1, ax1 = plt.subplots()
                ax1.scatter(rules["support"], rules["confidence"])
                ax1.set_xlabel("Support")
                ax1.set_ylabel("Confidence")
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots()
                ax2.hist(rules["lift"], bins=20)
                ax2.set_xlabel("Lift")
                st.pyplot(fig2)

    else:
        st.info("Upload CSV file")


