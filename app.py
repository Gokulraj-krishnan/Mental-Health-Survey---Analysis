import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Depression Prediction App",
    layout="centered"
)

st.title("Mental Health Depression Prediction")
st.write("This app predicts depression risk using a trained Deep Learning model.")

# -----------------------------
# Model Class
# -----------------------------
class DepressionNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# -----------------------------
# Load Files
# -----------------------------
@st.cache_resource
def load_model_files():
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/label_encoders.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    model = DepressionNet(len(feature_columns))
    model.load_state_dict(
        torch.load("models/depression_model.pth", map_location=torch.device("cpu"))
    )
    model.eval()

    return model, scaler, encoders, feature_columns

model, scaler, encoders, feature_columns = load_model_files()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Survey Details")

gender = st.selectbox("Gender", ["Male", "Female"])

age = st.number_input(
    "Age",
    min_value=10,
    max_value=100,
    value=25
)

city = st.text_input("City", "Chennai")

working_status = st.selectbox(
    "Working Professional or Student",
    ["Student", "Working Professional"]
)

profession = st.text_input("Profession", "Student")

academic_pressure = st.slider(
    "Academic Pressure",
    0.0,
    5.0,
    2.0
)

work_pressure = st.slider(
    "Work Pressure",
    0.0,
    5.0,
    2.0
)

cgpa = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0
)

study_satisfaction = st.slider(
    "Study Satisfaction",
    0.0,
    5.0,
    3.0
)

job_satisfaction = st.slider(
    "Job Satisfaction",
    0.0,
    5.0,
    3.0
)

sleep_duration = st.selectbox(
    "Sleep Duration",
    [
        "Less than 5 hours",
        "5-6 hours",
        "7-8 hours",
        "More than 8 hours"
    ]
)

dietary_habits = st.selectbox(
    "Dietary Habits",
    ["Healthy", "Moderate", "Unhealthy"]
)

degree = st.text_input("Degree", "B.Tech")

suicidal_thoughts = st.selectbox(
    "Have you ever had suicidal thoughts ?",
    ["No", "Yes"]
)

work_study_hours = st.slider(
    "Work/Study Hours",
    0,
    16,
    6
)

financial_stress = st.slider(
    "Financial Stress",
    0.0,
    5.0,
    2.0
)

family_history = st.selectbox(
    "Family History of Mental Illness",
    ["No", "Yes"]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Depression Risk"):

    input_data = {
        "Gender": gender,
        "Age": age,
        "City": city,
        "Working Professional or Student": working_status,
        "Profession": profession,
        "Academic Pressure": academic_pressure,
        "Work Pressure": work_pressure,
        "CGPA": cgpa,
        "Study Satisfaction": study_satisfaction,
        "Job Satisfaction": job_satisfaction,
        "Sleep Duration": sleep_duration,
        "Dietary Habits": dietary_habits,
        "Degree": degree,
        "Have you ever had suicidal thoughts ?": suicidal_thoughts,
        "Work/Study Hours": work_study_hours,
        "Financial Stress": financial_stress,
        "Family History of Mental Illness": family_history
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in input_df.columns:
        if col in encoders:
            le = encoders[col]

            try:
                input_df[col] = le.transform(input_df[col])
            except:
                input_df[col] = 0

    # Arrange columns same as training
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )
    # Encode categorical columns safely
    for col in input_df.columns:
        if input_df[col].dtype == "object":
            if col in encoders:
                le = encoders[col]
                input_df[col] = input_df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
            else:
                input_df[col] = 0
    # Scale input
    input_scaled = scaler.transform(input_df)

    # Convert to tensor
    input_tensor = torch.tensor(
        input_scaled,
        dtype=torch.float32
    )

    # Prediction
    with torch.no_grad():
        probability = model(input_tensor).item()

    st.subheader("Prediction Result")

    if probability >= 0.5:
        st.error("High Depression Risk")
    else:
        st.success("Low Depression Risk")

    st.write(f"Depression Probability: **{probability * 100:.2f}%**")