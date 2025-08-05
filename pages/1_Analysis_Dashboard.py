import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
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
tab1, tab2, tab3,tab4,tab_eff = st.tabs(["🔍 Drop-Off Points", "📈 Channel Comparison", "🧩 Funnel Analysis","🔄 Channel-to-Channel Flow","📈 Channel Conversion & Retention"])

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
    ax1.set_title("Drop-Off by Interaction Type")
    ax1.set_xlabel("Number of Interactions")
    ax1.set_ylabel("Interaction Type")
    st.pyplot(fig1)

# ───────────── Tab 2: Channel Comparison ─────
with tab2:
    st.header("📈 Channel Usage & Performance")
    st.markdown("Explore how each retail channel performs in terms of **total traffic** and **average engagement time**.")

    # ── Total Interactions by Channel ──
    st.subheader(" Total Interactions by Channel")
    ch_count = df["channel"].value_counts().reset_index()
    ch_count.columns = ["channel", "count"]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=ch_count, x="channel", y="count", palette="Blues_d", ax=ax2)
    ax2.set_title(" Total Interactions by Channel", fontsize=14)
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
    ax3.set_title("Avg Interaction Duration by Channel", fontsize=14)
    ax3.set_xlabel("Channel", fontsize=12)
    ax3.set_ylabel("Avg Duration (seconds)", fontsize=12)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha='right')
    for i, row in avg_duration.iterrows():
        ax3.text(i, row["duration"] + 0.5, f"{row['duration']:.1f}", ha='center', fontsize=10)
    st.pyplot(fig3)

# ───────────── Tab 3: Funnel ─────────────────
with tab3:
    st.header("🧩 Customer Funnel ")
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

# ────────────── Tab 4 – Channel Flow  (no interaction_time) ──────────
with tab4:
    st.header("🔄 Channel-to-Channel Flow")
    st.markdown("Visualizes how customers transition between channels over time.")

    # 1️⃣  Ensure rows are in the right order: customer → date → original order
    df_sorted = (
        df.sort_values(["customer_id", "interaction_date"])
          .reset_index(drop=True)
    )

    # 2️⃣  Get the next channel for each customer (date-order only)
    df_sorted["next_channel"] = (
        df_sorted.groupby("customer_id")["channel"].shift(-1)
    )

    # 3️⃣  Build the flow table
    flow = (df_sorted.dropna(subset=["next_channel"])
                      .groupby(["channel", "next_channel"])
                      .size()
                      .reset_index(name="count"))

    # 4️⃣  Keep only flows with meaningful volume (> 20 hops)
    flow = flow[flow["count"] > 20]

    # 5️⃣  Set node order by total traffic
    totals = (flow.groupby("channel")["count"].sum() +
              flow.groupby("next_channel")["count"].sum())
    node_order = totals.sort_values(ascending=False).index
    all_ch = pd.Index(node_order)

    # 6️⃣  Map channels to integer codes
    src = flow["channel"].apply(lambda x: all_ch.get_loc(x))
    tgt = flow["next_channel"].apply(lambda x: all_ch.get_loc(x))

    link_colors = src.map(lambda i: f"rgba({(i*60)%255},150,200,0.6)")

    sankey_fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=all_ch.to_list(), pad=20, thickness=20,
                  color="lightblue"),
        link=dict(source=src, target=tgt, value=flow["count"],
                  color=link_colors,
                  hovertemplate="%{source.label} ➜ %{target.label}"
                                "<br>%{value} hops<extra></extra>")
    ))
    sankey_fig.update_layout(height=500,
                             margin=dict(l=5, r=5, t=5, b=5))
    st.plotly_chart(sankey_fig, use_container_width=True)
# ─────────────────── Clean-channel mapping ───────────────────
channel_map = {
    "Physical Store": "Physical",
    "In Store Kiosk": "Physical",    

    "Web":    "Online",
    "Mobile": "Online",
    "Website":"Online"          
}

df["channel_clean"] = df["channel"].map(channel_map).fillna(df["channel"])

# ───────────── Tab 5 – Channel Effectiveness ─────────────
with tab_eff:
    st.header("📈 Channel Conversion & Retention")

    # ── 1️⃣  Conversion rate ------------------------------------------------
    conv_tbl = (df.groupby("channel_clean")["converted"]
                  .agg(visits="size",
                       purchases="sum",
                       conv_rate="mean")
                  .sort_values("conv_rate", ascending=False))

    st.subheader("🔑 Conversion Rate by Channel")
    st.dataframe(conv_tbl.style.format({"conv_rate": "{:.2%}"}))
    st.bar_chart(conv_tbl["conv_rate"])

    # χ² significance
    from scipy.stats import chi2_contingency
    cont = pd.crosstab(df["channel_clean"], df["converted"])
    chi2, p, *_ = chi2_contingency(cont)
    st.caption(f"χ² p-value = **{p:.4f}** "
               f"{'→ significant difference' if p < 0.05 else '→ no significant difference'}")

    st.markdown("---")

    # ── 2️⃣  365-day Retention ------------------------------------------------
    # build YYYY-MM-DD date from separate columns
    df["date"] = pd.to_datetime(
        dict(year=df["year"].astype(int),
             month=df["month"].astype(int),
             day=df["day"].astype(int)),
        errors="coerce"
    ).dt.normalize()

    first_date = df.groupby("customer_id")["date"].min()
    df["days_since_first"] = (df["date"] - df["customer_id"].map(first_date)).dt.days

    retained_flag = (df.groupby("customer_id")["days_since_first"]
                       .max()
                       .ge(365)      # came back ≥ 1 year later
                       .rename("retained"))

    overall_ret = retained_flag.mean()

    st.subheader("🔁 1-Year Retention")
    st.metric("Customers who returned ≥ 1 year later",
              f"{overall_ret*100:.1f}%")

    # channel-level retention (per customer × channel bucket)
    df["retained"] = df["customer_id"].map(retained_flag).astype(int)

    ch_ret = (df.groupby(["channel_clean", "customer_id"])["retained"]
                .max()
                .groupby("channel_clean")
                .mean()
                .sort_values(ascending=False))

    st.bar_chart(ch_ret)

