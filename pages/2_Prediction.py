import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page Config ──────────────────────────────
st.set_page_config(page_title="🎯 Purchase Prediction", layout="centered")
st.markdown("# 🎯 Predict Customer Purchase Behavior")
st.markdown("Use this form to simulate whether a customer will make a purchase based on their browsing behavior.")

# ── Load Model, Template, and Scaler ─────────
@st.cache_resource
def load_model():
    return joblib.load("xgml.joblib")

@st.cache_data
def load_template():
    return pd.read_csv("MLF.csv")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

# Load resources
model = load_model()
template_df = load_template()
scaler = load_scaler()

# ── Category Mapping ─────────────────────────
CATEGORY_CHOICES = [
    (0, "about_us"), (1, "account"), (2, "Area Rug"), (3, "Audio Equipment"),
    (4, "Bedding"), (5, "blog"), (6, "category_audio"), (7, "category_furniture"),
    (8, "category_gaming"), (9, "category_kitchen"), (10, "category_laptops"),
    (11, "category_smart_home"), (12, "category_smartphones"), (13, "category_tvs"),
    (14, "checkout"), (15, "Computer Accessories"), (16, "Cookware"),
    (17, "Desktop Computers"), (18, "Furniture"), (19, "Gaming"), (20, "home"),
    (21, "Kitchen Appliances"), (22, "Laptops"), (23, "order_history"),
    (24, "product_listing"), (25, "search_results"), (26, "Smart Home"),
    (27, "Smartphones"), (28, "store_locator"), (29, "support"),
    (30, "Tablets"), (31, "TVs"), (32, "wishlist"),
]
category_map = {label: code for code, label in CATEGORY_CHOICES}
reverse_map = {code: label for code, label in CATEGORY_CHOICES}

# ── Select Features to Use ───────────────────
required_features = model.feature_names_in_

# ── Custom Button Style ───────────────────────
st.markdown("""
    <style>
    .stButton>button {
        background-color: #00C49A;
        color: white;
        padding: 0.6em 1.2em;
        font-weight: bold;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ── Input Form ───────────────────────────────
user_inputs = {}
with st.form("prediction_form"):
    st.markdown("## 📝 Fill in Customer Details")

    st.markdown("### 👤 Customer Profile")
    for col in required_features:
        if col != "page_or_product":
            label = "⏱️ Time Spent in Session" if col == "duration" else f"{col.capitalize()} ⚙️"
            tooltip = f"Adjust the value for {col.replace('_', ' ')}"
            if template_df[col].dtype == "object" or template_df[col].nunique() <= 10:
                options = sorted(template_df[col].dropna().unique())
                user_inputs[col] = st.selectbox(label, options, help=tooltip)
            else:
                min_val = int(template_df[col].min())
                max_val = int(template_df[col].max())
                default = int(template_df[col].mean())
                user_inputs[col] = st.slider(label, min_val, max_val, default, help=tooltip)

    st.markdown("### 🧠 The Product They're Curious About")
    selected_label = st.selectbox(
        "What product/page caught their attention first?",
        [label for _, label in CATEGORY_CHOICES],
        help="This is where their curiosity began — the page or product they explored first."   
    )
    user_inputs["page_or_product"] = category_map[selected_label]  # ✅ Must be inside the form!

    submitted = st.form_submit_button("🔮 Will this customer purchase at the end of their journey?")

# ── Predict ──────────────────────────────────
if submitted:
    input_df = pd.DataFrame([user_inputs])
    input_df = input_df.reindex(columns=required_features)

    # Apply scaling
    scaler_features = scaler.feature_names_in_
    input_df[scaler_features] = scaler.transform(input_df[scaler_features])

    prediction = model.predict(input_df)[0]
    probas = model.predict_proba(input_df)[0]

    # ── Result Section ───────────────────────
    st.markdown("---")
    if prediction == 1:
        st.success(f"✅ **Will Purchase At The End** (Confidence: {probas[1]:.2%})")
    else:
        st.error(f"❌ **Will Not Purchase At The End** (Confidence: {probas[0]:.2%})")

    if probas[1] < 0.10:
        st.warning("⚠️ This customer seems highly unlikely to convert. Try tweaking inputs to test different outcomes.")

    st.markdown("### 📊 Prediction Breakdown")
    st.metric("🟢 Will Purchase", f"{probas[1]*100:.2f}%")
    st.metric("🔴 Will Not Purchase", f"{probas[0]*100:.2f}%")
