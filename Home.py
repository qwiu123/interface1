import streamlit as st

st.set_page_config(
    page_title="Multi-Channel Analytics Dashboard",
    layout="centered",
    page_icon="📈"
)

# Title and Subtitle
st.title("📦 Multi-Channel Customer Journey Analytics")
st.subheader("Enhancing Retail Customer Experience through Data")

# Intro Text
st.write("""
Welcome to the analytics dashboard. This system helps you:
- Analyze customer behavior across online and offline channels
- Identify dropout points in the purchase journey
- Predict if a customer will complete a purchase

---

📊 **Explore** the data on the **Data Analysis** page  
🤖 **Try out** the ML model on the **Prediction** page
""")

# Optional visual (placeholder for future image/diagram)
st.image("multichannel.jpg", width=1600)

# Optional Call to Action
st.success("Get started by choosing  ➡️")

st.page_link("pages/1_Analysis_Dashboard.py", label="📊 Go to Data Analysis", icon="📈")
st.page_link("pages/2_Prediction.py", label="🤖 Try Prediction Model", icon="🔮")
