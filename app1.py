import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ✅ Load Model & Scaler
def load_model():
    with open("stroke_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def load_scaler():
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    return scaler

model = load_model()
scaler = load_scaler()

# 🎨 **Set Page Config**
st.set_page_config(page_title="Heart Stroke Risk Assessment", page_icon="💖", layout="wide")

# 🎨 **Custom CSS for Improved UI**
custom_css = """
    <style>
    /* 🌟 Background Gradient */
    body, .stApp {
        background: linear-gradient(to right, #222, #b30000);
        color: #ffffff !important;
        font-size: 26px !important;
        text-align: center !important;
    }
    /* 📝 Headings */
    h1, h2, h3 {
        color: #ffffff !important;
        font-size: 44px !important;
        font-weight: bold !important;
        text-align: center !important;
    }
    /* 🎯 Change Sidebar (Navigation) Background to Black */
    section[data-testid="stSidebar"] {
        background-color: black !important;
        color: white !important;
    }
    /* 📌 Input Fields */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input, 
    .stSelectbox>div>div>select, 
    .stRadio>div, .stSlider>div>div>div {
        background-color: rgba(255, 255, 255, 0.2);
        color: white !important;
        border: 3px solid white !important;
        border-radius: 12px;
        font-size: 26px !important;
        padding: 12px !important;
    }
    /* 🎯 Make Input Labels White */
    label {
        color: white !important;
        font-size: 28px !important;
        font-weight: bold !important;
    }
    /* ✅ "Yes" and "No" Text Clearly Visible */
    .stRadio div label {
        color: white !important;
        font-weight: bold !important;
        font-size: 28px !important;
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px 20px;
        border-radius: 8px;
        border: 2px solid white !important;
        display: inline-block;
        width: 120px;
        text-align: center;
    }

    /* 🎯 Style the "Check Stroke" Button */
    div.stButton > button {
        background-color: white !important;
        color: black !important;
        font-size: 28px !important;
        font-weight: bold !important;
        border-radius: 12px;
        padding: 12px 24px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #dddddd !important;
    }

    /* 🔴 Stroke Risk Circle */
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.9; }
        50% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(1); opacity: 0.9; }
    }
    .stroke-risk-circle {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: red;
        color: white;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        animation: pulse 2s infinite;
        margin: auto;
    }
    /* 🛑 Prevention Box */
    .prevention-box {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        font-size: 24px;
        text-align: left;
        color: #ffffff;
        margin-top: 20px;
        border-left: 6px solid yellow;
    }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 🎨 **Sidebar Navigation**
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2304/2304043.png", width=130)
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Select a Page", ["🏠 Home", "ℹ️ About Stroke"])

# 📌 **Home Page - Prediction Interface**
if page == "🏠 Home":
    st.markdown("<h1>💓 Stroke Risk Assessment</h1>", unsafe_allow_html=True)
    st.markdown("### 🏥 Please enter your details below to assess your risk of stroke.")

    # 🎨 **Input Form**
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("👤 **Biological Sex**", ["Male", "Female"])
        age = st.slider("📅 **Your Age (Years)**", 20, 90, 50, step=1, format="%d")
        
        # Smoker selection
        smoker = st.radio("🚬 **Do you smoke?**", ["No", "Yes"])
        cigsPerDay = None

        # Show "Cigarettes per Day" only if "Smokes" is selected
        if smoker == "Yes":
            cigsPerDay = st.slider("🚬 **Number of cigarettes per day**", 1, 40, 10, step=1, format="%d")

        BPMeds = st.radio("💊 **Are you on blood pressure medication?**", ["No", "Yes"])
        prevalentStroke = st.radio("⚠️ **Have you previously had a stroke?**", ["No", "Yes"])

    with col2:
        prevalentHyp = st.radio("🔴 **Do you have high blood pressure?**", ["No", "Yes"])
        diabetes = st.radio("🍬 **Do you have diabetes?**", ["No", "Yes"])
        totChol = st.number_input("🩸 **Your total cholesterol level (mg/dL)**", min_value=100, max_value=600, value=200, step=5)
        sysBP = st.number_input("💓 **Your systolic blood pressure (mmHg)**", min_value=80, max_value=250, value=120, step=5)
        diaBP = st.number_input("💓 **Your diastolic blood pressure (mmHg)**", min_value=50, max_value=150, value=80, step=5)
        BMI = st.number_input("⚖️ **Your Body Mass Index (BMI)**", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        heartRate = st.number_input("❤️ **Your resting heart rate (BPM)**", min_value=40, max_value=150, value=75, step=1)
        glucose = st.number_input("🍭 **Your blood glucose level (mg/dL)**", min_value=50, max_value=300, value=100, step=5)

    # 🎯 **Prediction Button**
    if st.button("🔍 **Check Stroke Risk**"):
        # ✅ Generate Risk Percentage
        probability = np.random.uniform(0, 1)  # Simulating prediction
        risk_percentage = int(probability * 100)

        # 🎯 **Display Stroke Risk Circle**
        st.markdown(f"<div class='stroke-risk-circle'>Risk: {risk_percentage}%</div>", unsafe_allow_html=True)

        # 🛑 **Prevention Measures Based on Risk Level**
        st.markdown("### 🛡️ Prevention & Measures")

        if risk_percentage < 30:
            st.markdown("<div class='prevention-box'>✅ **Low Risk:** Maintain a healthy diet, exercise, and monitor health.</div>", unsafe_allow_html=True)
        elif risk_percentage < 60:
            st.markdown("<div class='prevention-box'>⚠️ **Moderate Risk:** Reduce salt, avoid smoking, exercise daily, and have regular checkups.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='prevention-box'>🚨 **High Risk:** Stop smoking, control blood pressure, manage diabetes, and consult a doctor immediately.</div>", unsafe_allow_html=True)
