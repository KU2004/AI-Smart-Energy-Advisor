import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Energy Advisor", layout="wide")

# ---------------- ADVANCED UI STYLE ----------------
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Card style */
.card {
    background: rgba(255, 255, 255, 0.08);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    text-align: center;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #22c55e, #06b6d4);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
    border: none;
}

/* Titles */
h1, h2, h3 {
    color: #f1f5f9;
}

/* Divider spacing */
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("⚡ Smart Energy Advisor")
st.markdown("#### 💡 Simple • Smart • Beautiful Energy Insights")

st.divider()

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload your electricity data", type=["txt", "csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file, sep=';', low_memory=False, na_values=['?'])
    st.success("✅ Data uploaded successfully")

    # ---------------- PREPROCESS ----------------
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df = df.dropna()

    df = df.rename(columns={'Global_active_power': 'energy'})
    df = df[['timestamp', 'energy']]

    df = df.sample(20000)

    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month

    X = df[['hour', 'day', 'month']]
    y = df['energy']

    st.divider()

    if st.button("🚀 Analyze My Energy"):

        # MODEL
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = RandomForestRegressor(n_estimators=50)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        avg_usage = np.mean(predictions)
        max_usage = df['energy'].max()
        min_usage = df['energy'].min()

        # ---------------- KPI SECTION ----------------
        st.subheader("📊 Energy Overview")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f'<div class="card">⚡<br><b>Average</b><br>{avg_usage:.2f}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="card">🔥<br><b>Highest</b><br>{max_usage:.2f}</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="card">🌙<br><b>Lowest</b><br>{min_usage:.2f}</div>', unsafe_allow_html=True)

        st.divider()

        # ---------------- GRAPH ----------------
        st.subheader("📈 Usage Trend")

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(y_test.values, label="Actual")
        ax.plot(predictions, label="Expected", linestyle='dashed')
        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)

        st.divider()

        # ---------------- STATUS ----------------
        st.subheader("📌 Your Status")

        if avg_usage > 600:
            st.error("⚠️ High Usage — Reduce load to save cost")
            status = "HIGH"
        elif avg_usage < 300:
            st.success("✅ Low Usage — Great energy saving")
            status = "LOW"
        else:
            st.info("ℹ️ Normal Usage — Balanced consumption")
            status = "NORMAL"

        st.divider()

        # ---------------- ACTION PLAN ----------------
        st.subheader("🎯 Smart Action Plan")

        if status == "HIGH":
            st.warning("👉 Reduce evening usage")
            st.warning("👉 Turn off unused appliances")
            st.warning("👉 Avoid running heavy devices together")
        elif status == "NORMAL":
            st.info("👉 Slight optimization can reduce cost")
            st.info("👉 Monitor peak hours")
        else:
            st.success("👉 Excellent usage pattern")
            st.success("👉 Continue same habits")

        st.divider()

        # ---------------- BEST TIME ----------------
        st.subheader("⏰ Best Time to Use Electricity")

        hourly = df.groupby('hour')['energy'].mean()
        best_hour = hourly.idxmin()
        worst_hour = hourly.idxmax()

        col1, col2 = st.columns(2)

        col1.success(f"✅ Best Time: {best_hour}:00")
        col2.error(f"⚠️ Avoid Time: {worst_hour}:00")

        st.divider()

        # ---------------- DECISION HELPER ----------------
        st.subheader("🧠 Quick Decision Helper")

        option = st.selectbox(
            "When do you want to use electricity?",
            ["Morning ☀️", "Afternoon 🌤️", "Evening 🌆", "Night 🌙"]
        )

        mapping = {
            "Morning ☀️": 9,
            "Afternoon 🌤️": 14,
            "Evening 🌆": 19,
            "Night 🌙": 23
        }

        selected_hour = mapping[option]

        future = model.predict([[selected_hour, 15, 6]])

        st.metric("⚡ Expected Usage", f"{future[0]:.2f}")

        if future[0] > 600:
            st.error("⚠️ Not a good time")
        else:
            st.success("✅ Good time")

        st.divider()

        # ---------------- DAILY PLAN ----------------
        st.subheader("📅 Daily Energy Guide")

        st.markdown("""
        - ☀️ Morning → Low usage ✅  
        - 🌤️ Afternoon → Moderate ⚖️  
        - 🌆 Evening → High ⚠️  
        - 🌙 Night → Low ✅  
        """)

        st.divider()

        # ---------------- ALERT ----------------
        st.subheader("🚨 Energy Alert")

        if avg_usage > 650:
            st.error("🚨 Very high usage! Act now.")
        elif avg_usage > 500:
            st.warning("⚠️ Usage increasing")
        else:
            st.success("✅ Everything is fine")

        st.divider()

        # ---------------- REPORT ----------------
        st.subheader("🧾 Download Report")

        report = f"""
SMART ENERGY REPORT

Average Usage: {avg_usage:.2f}
Status: {status}

Best Time: {best_hour}:00
Avoid Time: {worst_hour}:00
"""

        st.download_button("📥 Download Report", report, "energy_report.txt")

else:
    st.info("📂 Upload your dataset to begin")