import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load the trained model
def load_model():
    if os.path.exists("stroke_model.pkl"):
        with open("stroke_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    else:
        st.error("❌ Model file 'stroke_model.pkl' not found! Train and save the model first.")
        st.stop()

# Load the encoder dictionary
def load_encoders():
    if os.path.exists("vector.pkl"):
        with open("vector.pkl", "rb") as file:
            return pickle.load(file)  # This will return a dictionary of LabelEncoders
    else:
        st.error("❌ Encoder file 'vector.pkl' not found! Train and save the encoder first.")
        st.stop()

# Define categorical columns
categorical_cols = ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]

# ✅ Load model and encoder
model = load_model()
encoders = load_encoders()

# Streamlit UI
st.title("Heart Stroke Prediction App")

# Input fields for user
age = st.slider("Age", 0, 100, 50)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
disease = st.selectbox("Heart Disease", ["No", "Yes"])
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt job", "Children", "Never worked"])
residence = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose = st.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0)
smoking = st.selectbox("Smoking Status", ["Never smoked", "Formerly smoked", "Smokes", "Unknown"])

# Create DataFrame for input
input_data = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "hypertension": [1 if hypertension == "Yes" else 0],
    "heart_disease": [1 if disease == "Yes" else 0],
    "ever_married": [married],
    "work_type": [work_type],
    "residence_type": [residence],
    "avg_glucose_level": [avg_glucose],
    "bmi": [bmi],
    "smoking_status": [smoking]
})

# 🔹 Standardize column names
input_data.columns = input_data.columns.str.lower().str.strip()

# ✅ Apply encoding to categorical columns
for col in categorical_cols:
    if col in encoders:
        input_data[col] = encoders[col].transform([input_data[col][0]])[0]  # Convert category to number
    else:
        st.error(f"❌ Encoder for {col} not found in 'vector.pkl'.")
        st.stop()

# Convert to NumPy array
input_data_encoded = np.array(input_data).reshape(1, -1)  # Ensure correct shape for model

# ✅ Make prediction
if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_data_encoded)
    probability = model.predict_proba(input_data_encoded)
    print("Raw Prediction:", prediction)
    print("Prediction Probability:", probability)  # Debugging
    if prediction[0] == 1:
        st.error("⚠️ High risk of stroke! Consult a doctor.")
    else:
        st.success("✅ Low risk of stroke! Keep maintaining a healthy lifestyle.")
