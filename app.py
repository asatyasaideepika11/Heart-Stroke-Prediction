import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ✅ Load the trained model & scaler
def load_model():
    with open("stroke_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def load_scaler():
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    return scaler

# Load model and scaler
model = load_model()
scaler = load_scaler()

# 🎨 Custom Page Styling
st.set_page_config(page_title="Heart Stroke Prediction", page_icon="💓", layout="wide")

# 🎨 Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2304/2304043.png", width=120)
st.sidebar.title("🔍 Navigate")
page = st.sidebar.radio("Go to", ["Home", "About Stroke"])

# 📌 Home Page - Prediction Interface
if page == "Home":
    st.markdown("<h1 style='text-align: center; color: red;'>💓 Heart Stroke Prediction</h1>", unsafe_allow_html=True)
    st.markdown("### 🏥 Enter Your Health Details Below:")
    
    # 🎨 UI Layout
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        age = st.slider("📅 Age", 20, 90, 50)
        smoker = st.radio("🚬 Current Smoker?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        cigsPerDay = st.slider("🚬 Cigarettes Per Day", 0, 40, 10)
        BPMeds = st.radio("💊 On Blood Pressure Medication?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        prevalentStroke = st.radio("⚠️ Previous Stroke?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    with col2:
        prevalentHyp = st.radio("🔴 Hypertension?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        diabetes = st.radio("🍬 Diabetes?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        totChol = st.number_input("🩸 Total Cholesterol", min_value=100, max_value=600, value=200)
        sysBP = st.number_input("💓 Systolic BP", min_value=80, max_value=250, value=120)
        diaBP = st.number_input("💓 Diastolic BP", min_value=50, max_value=150, value=80)
        BMI = st.number_input("⚖️ Body Mass Index", min_value=10.0, max_value=50.0, value=25.0)
        heartRate = st.number_input("❤️ Heart Rate", min_value=40, max_value=150, value=75)
        glucose = st.number_input("🍭 Glucose Level", min_value=50, max_value=300, value=100)

    # 🎯 Prediction Button
    if st.button("🔍 Predict Stroke Risk"):
        # ✅ Prepare input data
        input_data = pd.DataFrame({
            "male": [1 if gender == "Male" else 0],
            "age": [age],
            "currentSmoker": [smoker],
            "cigsPerDay": [cigsPerDay],
            "BPMeds": [BPMeds],
            "prevalentStroke": [prevalentStroke],
            "prevalentHyp": [prevalentHyp],
            "diabetes": [diabetes],
            "totChol": [totChol],
            "sysBP": [sysBP],
            "diaBP": [diaBP],
            "BMI": [BMI],
            "heartRate": [heartRate],
            "glucose": [glucose]
        })

        # ✅ Normalize numerical features
        input_data[["totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]] = scaler.transform(
            input_data[["totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]]
        )

        # ✅ Make Prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]  # Get probability of stroke

        # 🎯 Display Results
        st.markdown("## 🏥 Prediction Result")
        progress = int(probability * 100)

        if probability >= 0.30:  # High Stroke Risk
            st.error(f"⚠️ **High Risk of Stroke!** (Confidence: {probability:.2f})")
            st.progress(progress)
            st.markdown("### 🔴 **Health Recommendations:**")
            st.write("- 🏃 Engage in daily physical activity")
            st.write("- 🥦 Follow a balanced diet")
            st.write("- 🚭 Quit smoking immediately")
            st.write("- 💊 Manage blood pressure & diabetes")
        else:  # Low Stroke Risk
            st.success(f"✅ **Low Risk of Stroke!** (Confidence: {1 - probability:.2f})")
            st.progress(progress)
            st.markdown("### 🟢 **Health Tips:**")
            st.write("- 💪 Maintain a healthy lifestyle")
            st.write("- 🚶 Walk at least 30 minutes a day")
            st.write("- 🍏 Eat more fruits & vegetables")
            st.write("- ❤️ Keep stress levels low")

# 📌 About Stroke Page
elif page == "About Stroke":
    st.markdown("## 🔬 What is Stroke?")
    st.write("A stroke happens when blood flow to the brain is disrupted, causing brain cells to die.")
    st.image("https://www.stroke.org.uk/sites/default/files/styles/large/public/what_is_stroke_0.jpg", width=700)
    st.markdown("### ⚠️ Risk Factors:")
    st.write("- **High Blood Pressure** (Hypertension)")
    st.write("- **Smoking & Alcohol Consumption**")
    st.write("- **Diabetes & High Cholesterol**")
    st.write("- **Obesity & Lack of Physical Activity**")
    st.write("- **Heart Disease**")
    st.write("- **Family History**")

    st.markdown("### 🏥 **How to Prevent Stroke?**")
    st.write("- 🚶 Exercise regularly")
    st.write("- 🥗 Eat a healthy diet")
    st.write("- 🏋️ Maintain a healthy weight")
    st.write("- 💊 Control blood pressure & diabetes")
    st.write("- 🚭 Stop smoking & reduce alcohol")

