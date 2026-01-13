import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="RFM Evaluation Metrics", layout="wide")
st.title("📊 RFM Analysis – Evaluation Metrics")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file is None:
    st.info("👆 Upload a CSV file to start RFM analysis")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# =========================
# RFM CALCULATION
# =========================
required_cols = ["CustomerID", "InvoiceDate", "InvoiceID", "TotalAmount"]
if not all(col in df.columns for col in required_cols):
    st.error(f"Dataset must contain columns: {required_cols}")
    st.stop()

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceID": "count",
    "TotalAmount": "sum"
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

# =========================
# SCALING
# =========================
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
rfm[["Recency_Scaled", "Frequency_Scaled", "Monetary_Scaled"]] = rfm_scaled

# =========================
# RFM SCORING
# =========================
rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1])
rfm["F_Score"] = pd.qcut(rfm["Frequency"], 5, labels=[1,2,3,4,5])
rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5])

rfm["RFM_Total"] = (
    rfm["R_Score"].astype(int) +
    rfm["F_Score"].astype(int) +
    rfm["M_Score"].astype(int)
)

st.subheader("📌 RFM Table")
st.dataframe(rfm.head())

# =========================
# EVALUATION METRICS
# =========================
st.header("📈 RFM Evaluation Metrics")

# 1. Descriptive Statistics
st.subheader("1️⃣ Descriptive Statistics")
st.dataframe(rfm[["Recency", "Frequency", "Monetary"]].describe())

# 2. Coefficient of Variation
st.subheader("2️⃣ Coefficient of Variation (Customer Variability)")
cv = rfm[["Recency", "Frequency", "Monetary"]].std() / rfm[["Recency", "Frequency", "Monetary"]].mean()
st.write(cv)

# 3. Skewness
st.subheader("3️⃣ Skewness (Distribution Shape)")
st.write(rfm[["Recency", "Frequency", "Monetary"]].skew())

# 4. Kurtosis
st.subheader("4️⃣ Kurtosis (Outlier Detection)")
st.write(rfm[["Recency", "Frequency", "Monetary"]].kurtosis())

# 5. RFM Segment Distribution
st.subheader("5️⃣ RFM Segment Distribution (%)")
segment_dist = rfm["RFM_Total"].value_counts(normalize=True) * 100
st.write(segment_dist.sort_index())

# 6. Pareto (80/20 Rule)
st.subheader("6️⃣ Pareto Analysis (80/20 Rule)")
rfm_sorted = rfm.sort_values("Monetary", ascending=False)
revenue_top_20 = (
    rfm_sorted.iloc[:int(0.2 * len(rfm_sorted))]["Monetary"].sum()
    / rfm_sorted["Monetary"].sum()
)
st.metric("Revenue from Top 20% Customers", f"{revenue_top_20:.2%}")

# 7. RFM Consistency
st.subheader("7️⃣ RFM Consistency Metric")
rfm_consistency = rfm[["RFM_Total", "Monetary"]].corr().iloc[0,1]
st.metric("Correlation (RFM Score vs Monetary)", f"{rfm_consistency:.3f}")

# =========================
# VISUAL EVALUATION
# =========================
st.header("📊 Visual Evaluation")

# Correlation Heatmap
fig, ax = plt.subplots()
sns.heatmap(
    rfm[["Recency", "Frequency", "Monetary"]].corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)
ax.set_title("RFM Feature Correlation Matrix")
st.pyplot(fig)

# Histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(rfm["Recency"], bins=30, ax=axes[0])
axes[0].set_title("Recency Distribution")
sns.histplot(rfm["Frequency"], bins=30, ax=axes[1])
axes[1].set_title("Frequency Distribution")
sns.histplot(rfm["Monetary"], bins=30, ax=axes[2])
axes[2].set_title("Monetary Distribution")
st.pyplot(fig)

# RFM Score Distribution
fig, ax = plt.subplots()
rfm["RFM_Total"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_xlabel("RFM Total Score")
ax.set_ylabel("Customers")
ax.set_title("Customer Distribution by RFM Score")
st.pyplot(fig)

# Scaled Feature Space
fig, ax = plt.subplots()
ax.scatter(rfm["Recency_Scaled"], rfm["Monetary_Scaled"], alpha=0.6)
ax.set_xlabel("Recency (Scaled)")
ax.set_ylabel("Monetary (Scaled)")
ax.set_title("Scaled RFM Feature Space")
st.pyplot(fig)
