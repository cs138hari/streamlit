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

        # ---------- RFM Calculation ----------
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

        # ---------- Scaling ----------
        scaler = StandardScaler()
        rfm[["R_Scaled", "F_Scaled", "M_Scaled"]] = scaler.fit_transform(
            rfm[["Recency", "Frequency", "Monetary"]]
        )

        # ---------- RFM Scores ----------
        rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
        rfm["F_Score"] = pd.qcut(rfm["Frequency"], 5, labels=[1, 2, 3, 4, 5])
        rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

        rfm["RFM_Total"] = (
            rfm["R_Score"].astype(int)
            + rfm["F_Score"].astype(int)
            + rfm["M_Score"].astype(int)
        )

        # ================= RFM EVALUATION METRICS =================
        st.subheader("📈 RFM Evaluation Metrics")

        # 1. Descriptive Statistics
        st.markdown("### 1️⃣ Descriptive Statistics")
        st.dataframe(rfm[["Recency", "Frequency", "Monetary"]].describe())

        # 2. Coefficient of Variation
        st.markdown("### 2️⃣ Coefficient of Variation (Customer Variability)")
        cv = rfm[["Recency", "Frequency", "Monetary"]].std() / \
             rfm[["Recency", "Frequency", "Monetary"]].mean()
        st.dataframe(cv.to_frame("Coefficient of Variation"))

        # 3. Skewness
        st.markdown("### 3️⃣ Skewness (Distribution Shape)")
        st.dataframe(rfm[["Recency", "Frequency", "Monetary"]].skew().to_frame("Skewness"))

        # 4. Kurtosis
        st.markdown("### 4️⃣ Kurtosis (Outlier Detection)")
        st.dataframe(rfm[["Recency", "Frequency", "Monetary"]].kurtosis().to_frame("Kurtosis"))

        # 5. RFM Segment Distribution
        st.markdown("### 5️⃣ RFM Segment Distribution (%)")
        segment_dist = rfm["RFM_Total"].value_counts(normalize=True) * 100
        st.dataframe(segment_dist.to_frame("Percentage"))

        # 6. Pareto Analysis
        st.markdown("### 6️⃣ Pareto Analysis (80/20 Rule)")
        rfm_sorted = rfm.sort_values("Monetary", ascending=False)
        revenue_top_20 = (
            rfm_sorted.iloc[:int(0.2 * len(rfm_sorted))]["Monetary"].sum()
            / rfm_sorted["Monetary"].sum()
        )
        st.metric("Revenue from Top 20% Customers", f"{revenue_top_20:.2%}")

        # 7. RFM Consistency
        st.markdown("### 7️⃣ RFM Consistency Metric")
        rfm_consistency = rfm[["RFM_Total", "Monetary"]].corr().iloc[0, 1]
        st.metric("Correlation (RFM Score vs Monetary)", f"{rfm_consistency:.3f}")

        # ---------- Visualizations ----------
        st.subheader("📊 RFM Distributions")
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(rfm["Recency"], bins=30, ax=ax[0])
        sns.histplot(rfm["Frequency"], bins=30, ax=ax[1])
        sns.histplot(rfm["Monetary"], bins=30, ax=ax[2])
        st.pyplot(fig)

        st.subheader("📊 RFM Score Distribution")
        fig2, ax2 = plt.subplots()
        rfm["RFM_Total"].value_counts().sort_index().plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

    else:
        st.info("Upload CSV file to perform RFM analysis")

# =====================================================
# TAB 2: K-MEANS CLUSTERING (MODIFIED: K = 0–10 DISPLAY)
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

        # ---------- Elbow Method (K = 0–10 on X-axis) ----------
        wcss = []
        K_RANGE = range(0, 11)

        for k_val in K_RANGE:
            if k_val < 2:
                wcss.append(np.nan)   # invalid K values
            else:
                km = KMeans(
                    n_clusters=k_val,
                    random_state=42,
                    n_init=10
                )
                km.fit(scaled_data)
                wcss.append(km.inertia_)

        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(K_RANGE, wcss, marker="o")
        ax_elbow.set_xlabel("Number of Clusters (K)")
        ax_elbow.set_ylabel("WCSS")
        ax_elbow.set_title("Elbow Method (K = 0–10)")
        st.pyplot(fig_elbow)

        # ---------- Select K (VALID RANGE ONLY) ----------
     k = st.slider(
    "Select number of clusters (K)",
    min_value=0,
    max_value=10,
    value=3
)

if k < 2:
    st.warning("⚠️ K-Means clustering requires K ≥ 2. Please select K = 2 or higher.")
    st.stop()

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)
clusters = kmeans.fit_predict(scaled_data)
df["KMeans_Cluster"] = clusters


        # ---------- Evaluation Metrics ----------
        st.subheader("📊 K-Means Evaluation Metrics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Inertia (WCSS)", f"{kmeans.inertia_:.2f}")
        col2.metric(
            "Silhouette Score",
            f"{silhouette_score(scaled_data, clusters):.4f}"
        )
        col3.metric(
            "Davies–Bouldin Index",
            f"{davies_bouldin_score(scaled_data, clusters):.4f}"
        )
        col4.metric(
            "Calinski–Harabasz Index",
            f"{calinski_harabasz_score(scaled_data, clusters):.2f}"
        )

        # ---------- Cluster Visualization ----------
        st.subheader("📊 Cluster Visualization")

        fig_cluster, ax_cluster = plt.subplots(figsize=(8, 6))
        ax_cluster.scatter(
            scaled_data[:, 0],
            scaled_data[:, 1],
            c=clusters,
            cmap="viridis"
        )
        ax_cluster.set_xlabel("Feature 1 (Scaled)")
        ax_cluster.set_ylabel("Feature 2 (Scaled)")
        ax_cluster.set_title("K-Means Clustering Result")
        st.pyplot(fig_cluster)

        # ---------- PCA Visualization ----------
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)

        fig_pca, ax_pca = plt.subplots()
        ax_pca.scatter(
            pca_data[:, 0],
            pca_data[:, 1],
            c=clusters,
            cmap="viridis"
        )
        ax_pca.set_xlabel("PCA 1")
        ax_pca.set_ylabel("PCA 2")
        ax_pca.set_title("K-Means Clusters (PCA View)")
        st.pyplot(fig_pca)

        # ---------- Cluster Distribution ----------
        st.subheader("📊 Cluster Size Distribution")
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        st.bar_chart(cluster_counts)

    else:
        st.info("Upload CSV file to perform K-Means clustering")


# =====================================================
# TAB 3: APRIORI
# =====================================================
# =====================================================
# TAB 3: APRIORI
# =====================================================
with tab3:
    if df is not None:
        st.header("🔗 Apriori Algorithm")

        basket = (
            df.groupby(["InvoiceID", "ItemName"])["Quantity"]
            .sum()
            .unstack()
            .fillna(0)
        )
        basket = (basket > 0).astype(int)

        freq = apriori(basket, min_support=0.005, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=0.3)

        if not rules.empty:
            st.dataframe(rules.head())

            # -------------------------------
            # 3️⃣ Apriori Evaluation Metrics
            # -------------------------------
            st.subheader("📊 Apriori Evaluation Metrics")

            evaluation_metrics = rules[
                [
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift",
                    "leverage",
                    "conviction"
                ]
            ]

            st.dataframe(evaluation_metrics.head(10))

            # -------------------------------
            # 4️⃣ Apriori Visualizations
            # -------------------------------
            st.subheader("📊 Apriori Visualizations")

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
        num_scaled = StandardScaler().fit_transform(
            df[["Quantity", "TotalAmount"]]
        )

        kmeans = KMeans(n_clusters=3, random_state=42)
        df["Cluster"] = kmeans.fit_predict(num_scaled)

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

        if freq_items.empty:
            st.warning("❌ No frequent itemsets found")

        else:
            rules = association_rules(
                freq_items,
                metric="confidence",
                min_threshold=0.1
            )

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

                # =====================================
                # 🔎 COMBINED EVALUATION METRICS
                # =====================================
                st.subheader("📊 Combined Model Evaluation Metrics")

                # ---------- K-Means Metrics ----------
                kmeans_metrics = {
                    "Inertia (WCSS)": kmeans.inertia_,
                    "Silhouette Score": silhouette_score(num_scaled, df["Cluster"]),
                    "Davies-Bouldin Index": davies_bouldin_score(num_scaled, df["Cluster"]),
                    "Calinski-Harabasz Index": calinski_harabasz_score(num_scaled, df["Cluster"])
                }

                # ---------- Apriori Metrics ----------
                apriori_metrics = {
                    "Average Support": rules["support"].mean(),
                    "Average Confidence": rules["confidence"].mean(),
                    "Average Lift": rules["lift"].mean(),
                    "Average Leverage": rules["leverage"].mean(),
                    "Average Conviction": rules["conviction"].mean()
                }

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🔵 K-Means Metrics")
                    for k, v in kmeans_metrics.items():
                        st.metric(k, f"{v:.4f}")

                with col2:
                    st.markdown("### 🟢 Apriori Metrics")
                    for k, v in apriori_metrics.items():
                        st.metric(k, f"{v:.4f}")

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







