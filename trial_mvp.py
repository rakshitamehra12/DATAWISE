import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import requests
import os
from dotenv import load_dotenv


# Environment

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page Configuration

st.set_page_config(page_title="AI-Driven Diabetes CDS", layout="wide")

st.markdown("""
<style>
.main { background-color: #f5f7f9; }
.stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Model Training

@st.cache_resource
def train_model():
    file_path = "datasetdiabetes.csv"

    if not os.path.exists(file_path):
        st.error("Dataset not found. Please place datasetdiabetes.csv in project folder.")
        st.stop()

    data = pd.read_csv(file_path)

    gender_enc = LabelEncoder()
    smoking_enc = LabelEncoder()

    data["gender"] = gender_enc.fit_transform(data["gender"])
    data["smoking_history"] = smoking_enc.fit_transform(data["smoking_history"])

    X = data.drop("diabetes", axis=1)
    y = data["diabetes"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    return model, scaler, X.columns, gender_enc, smoking_enc

model, scaler, feature_names, gender_encoder, smoking_encoder = train_model()

# Sidebar Inputs

st.sidebar.header("📋 Patient Clinical Data")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
age = st.sidebar.slider("Age", 1, 100, 45)

col_a, col_b = st.sidebar.columns(2)
hypertension = col_a.selectbox("Hypertension", [0, 1])
heart_disease = col_b.selectbox("Heart Disease", [0, 1])

smoking = st.sidebar.selectbox("Smoking History", ["never", "former", "current", "No Info"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 24.5, step=0.1)
hba1c = st.sidebar.slider("HbA1c (%)", 4.0, 12.0, 5.5, step=0.1)
glucose = st.sidebar.slider("Blood Glucose (mg/dL)", 60, 350, 100)


# Risk Prediction

gender_val = gender_encoder.transform([gender])[0]
smoking_val = smoking_encoder.transform([smoking])[0]

input_data = pd.DataFrame([[
    gender_val, age, hypertension, heart_disease,
    smoking_val, bmi, hba1c, glucose
]], columns=feature_names)

scaled_input = scaler.transform(input_data)
risk_prob = model.predict_proba(scaled_input)[0][1]

if risk_prob < 0.3:
    risk_level = "Low"
elif risk_prob < 0.7:
    risk_level = "Moderate"
else:
    risk_level = "High"


# Dashboard

st.title("🩺 AI-Driven Diabetes Clinical Decision Support")

left, right = st.columns(2)

with left:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_prob * 100,
        title={"text": "Diabetes Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30], "color": "#c3e6cb"},
                {"range": [30, 70], "color": "#ffeeba"},
                {"range": [70, 100], "color": "#f8d7da"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Clinical Risk Interpretation")
    if risk_level == "Low":
        st.success("Low Risk – Maintain healthy lifestyle.")
    elif risk_level == "Moderate":
        st.warning("Moderate Risk – Lifestyle intervention advised.")
    else:
        st.error("High Risk – Diagnostic confirmation recommended.")


# Explainability

st.markdown("---")
st.subheader("🔍 Individual Risk Drivers")

contributions = model.coef_[0] * scaled_input[0]
contrib_df = pd.DataFrame({
    "Feature": feature_names,
    "Contribution": contributions
}).sort_values(by="Contribution", ascending=False)

top_factors = contrib_df.head(3)["Feature"].tolist()

fig2 = go.Figure(go.Bar(
    x=contrib_df["Contribution"],
    y=contrib_df["Feature"],
    orientation="h"
))
st.plotly_chart(fig2, use_container_width=True)


# AI FUNCTIONS 
def groq_call(prompt, system):
    if not GROQ_API_KEY:
        return "AI unavailable (API key missing)."

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model":"llama-3.1-8b-instant",  
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"AI service error: {response.text}"



st.markdown("---")
st.subheader("🤖 AI Lifestyle & Exercise Coach")

if st.button("Generate AI Health Tips"):
    with st.spinner("Generating recommendations..."):
        prompt = f"""
        Patient Risk: {risk_level}
        Age: {age}, BMI: {bmi}, HbA1c: {hba1c}, Glucose: {glucose}, Smoking: {smoking}
        Give diet, exercise and lifestyle guidance.
        """
        st.write(groq_call(prompt, "You are a preventive healthcare assistant."))

st.markdown("---")
st.subheader("🧠 AI Risk Explanation")

if st.button("Explain Risk in Simple Language"):
    with st.spinner("Explaining risk..."):
        prompt = f"""
        Risk Level: {risk_level}
        Key factors: {", ".join(top_factors)}
        Explain clearly to a patient.
        """
        st.write(groq_call(prompt, "You explain medical risk simply."))

st.markdown("---")
st.subheader("🧾 AI Doctor Summary")

if st.button("Generate Clinical Summary"):
    with st.spinner("Preparing summary..."):
        prompt = f"""
        Age: {age}, BMI: {bmi}, HbA1c: {hba1c}, Glucose: {glucose}
        Risk Level: {risk_level}
        Create a concise doctor summary with next steps.
        """
        summary = groq_call(prompt, "You write clinical summaries.")
        st.text_area("Doctor Review", summary, height=220)

st.caption("⚠️ Educational decision support only. Not a medical diagnosis.")
