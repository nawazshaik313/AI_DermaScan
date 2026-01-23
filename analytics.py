<<<<<<< HEAD
import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="DermalScan Analytics",
    page_icon="📊",
    layout="wide"
)

st.markdown("## 📊 DermalScan System Analytics")

LOG_FILE = "inference_logs.csv"

if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧪 Condition Distribution")
        st.bar_chart(df["Prediction"].value_counts())

    with col2:
        st.markdown("### 📈 Confidence Trend Over Time")
        df["Confidence"] = df["Confidence"].astype(float)
        st.line_chart(df["Confidence"])

    st.markdown("### 🧾 Full Inference Logs")
    st.dataframe(df, use_container_width=True)

else:
    st.warning("No inference logs found yet.")
=======
import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="DermalScan Analytics",
    page_icon="📊",
    layout="wide"
)

st.markdown("## 📊 DermalScan System Analytics")

LOG_FILE = "inference_logs.csv"

if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧪 Condition Distribution")
        st.bar_chart(df["Prediction"].value_counts())

    with col2:
        st.markdown("### 📈 Confidence Trend Over Time")
        df["Confidence"] = df["Confidence"].astype(float)
        st.line_chart(df["Confidence"])

    st.markdown("### 🧾 Full Inference Logs")
    st.dataframe(df, use_container_width=True)

else:
    st.warning("No inference logs found yet.")
>>>>>>> 46dc3471f383589b98ed4847bfbe1a9151001303
