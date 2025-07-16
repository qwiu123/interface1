import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ── Page Config ──────────────────────────────
st.set_page_config(page_title="📊 Customer Journey Analytics", layout="wide")
st.title("📊 Multi-Channel Customer Journey Analysis")

# ── Load Data ────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("interface.csv")

df = load_data()

# ── Prepare Converted Column ─────────────────
df["converted"] = (df["interaction_type"] == "Purchase Page").astype(int)

# ── Top KPIs ─────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("🧍‍♂️ Total Unique Customers", df["customer_id"].nunique())
col2.metric("⏱️ Avg Duration (s)", round(df["duration"].mean(), 1))
col3.metric("✅ Conversion Rate", f"{df['converted'].mean()*100:.2f}%")

st.markdown("---")

# ── Tabs ─────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Drop-Off Points", "📈 Channel Comparison", "🧩 Funnel Analysis"])

# ───────────── Tab 1: Drop-Off ──────────────
with tab1:
    st.header("🔍 Where Do Customers Drop Off?")
    st.markdown("This analysis highlights which stages customers frequently abandon before making a purchase.")

    non_purchase_df = df[df["interaction_type"] != "Purchase Page"]
    dropoff_counts = non_purchase_df["interaction_type"].value_counts().reset_index()
    dropoff_counts.columns = ["interaction_type", "count"]
    dropoff_counts = dropoff_counts.sort_values(by="count", ascending=True)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=dropoff_counts, y="interaction_type", x="count", palette="pastel", ax=ax1)
    ax1.set_title("Drop-Off by Interaction Type (Excludes Purchases)")
    ax1.set_xlabel("Number of Interactions")
    ax1.set_ylabel("Interaction Type")
    st.pyplot(fig1)

# ───────────── Tab 2: Channel Comparison ─────
with tab2:
    st.header("📈 Channel Usage & Performance")
    st.markdown("Explore how each retail channel performs in terms of **total traffic** and **average engagement time**.")

    # ── Total Interactions by Channel ──
    st.subheader("🧾 Total Interactions by Channel")
    ch_count = df["channel"].value_counts().reset_index()
    ch_count.columns = ["channel", "count"]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=ch_count, x="channel", y="count", palette="Blues_d", ax=ax2)
    ax2.set_title("🧾 Total Interactions by Channel", fontsize=14)
    ax2.set_xlabel("Channel", fontsize=12)
    ax2.set_ylabel("Number of Interactions", fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')
    for i, row in ch_count.iterrows():
        ax2.text(i, row["count"] + 1000, f"{int(row['count']):,}", ha='center', fontsize=10)
    st.pyplot(fig2)

    st.markdown("---")

    # ── Avg Duration by Channel ──
    st.subheader("⏱️ Average Interaction Duration by Channel")
    avg_duration = df.groupby("channel")["duration"].mean().reset_index()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=avg_duration, x="channel", y="duration", palette="coolwarm", ax=ax3)
    ax3.set_title("⏱️ Avg Interaction Duration by Channel", fontsize=14)
    ax3.set_xlabel("Channel", fontsize=12)
    ax3.set_ylabel("Avg Duration (seconds)", fontsize=12)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha='right')
    for i, row in avg_duration.iterrows():
        ax3.text(i, row["duration"] + 0.5, f"{row['duration']:.1f}", ha='center', fontsize=10)
    st.pyplot(fig3)

# ───────────── Tab 3: Funnel ─────────────────
with tab3:
    st.header("🧩 Customer Funnel (Sequential Path Only)")
    st.markdown("Tracks only customers who completed each step in order: Product View → Add to Cart → Checkout → Purchase.")

    stages = ["Product View", "Add To Cart", "Checkout", "Purchase Page"]
    funnel_data = df[df["interaction_type"].isin(stages)].copy()

    journey = funnel_data.groupby("customer_id")["interaction_type"].apply(list).reset_index()

    def check_steps(path):
        result = {s: 0 for s in stages}
        seen = set()
        for s in path:
            seen.add(s)
            for stage in stages:
                if all(prev in seen for prev in stages[:stages.index(stage)+1]):
                    result[stage] = 1
        return pd.Series(result)

    flags = journey["interaction_type"].apply(check_steps)
    counts = flags.sum().reindex(stages)

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.values, y=counts.index, palette="viridis", ax=ax4)
    ax4.set_title("Sequential Customer Funnel (Unique Customers)")
    ax4.set_xlabel("Number of Customers")
    ax4.set_ylabel("Funnel Stage")

    for i, val in enumerate(counts.values):
        ax4.text(val + 30, i, f"{val:,}", va="center")

    st.pyplot(fig4)

