import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Diabetes Risk App", layout="wide")

st.title("Diabetes Clinical Decision Support System")

# Sidebar #

st.sidebar.header("Patient Information")
api_key = st.sidebar.text_input("Optional: Enter Groq API Key", type="password")

# Load Dataset  #

@st.cache_data
def load_data():
    return pd.read_csv("datasetdiabetes.csv")

df = load_data()

# Show columns quietly for debugging (can remove later)
st.sidebar.write("Detected Columns:")
st.sidebar.write(list(df.columns))

def find_column(keyword):
    for col in df.columns:
        if keyword.lower() in col.lower():
            return col
    return None

target_col = find_column("diabetes") or find_column("outcome")

if target_col is None:
    st.error("Target column not found in dataset.")
    st.stop()

df_model = df.copy()  # creating a copy to avoid modifying original dataframe

# Encode categorical columns
encoders = {}

for col in df_model.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    encoders[col] = le

X = df_model.drop(columns=[target_col])
y = df_model[target_col]

# Train simple model
model = LogisticRegression(max_iter=1000, solver="lbfgs")
model.fit(X, y)

#  User Input Section  #

st.sidebar.subheader("Enter Clinical Values")

user_data = {}

for col in X.columns:
    if df[col].dtype == "object":
        user_data[col] = st.sidebar.selectbox(col, df[col].unique())
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        default = float(df[col].mean())
        user_data[col] = st.sidebar.slider(col, min_val, max_val, default)

input_df = pd.DataFrame([user_data]).copy()

# Apply same encoding to input
for col in input_df.columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

#  Prediction  #

risk = model.predict_proba(input_df)[0][1] * 100

st.subheader("Predicted Diabetes Risk")
st.metric("Risk Percentage", f"{risk:.2f}%")

# Basic Explanation#

def basic_advice(score):
    if score < 30:
        return """
Low risk.  
Try maintaining a balanced diet, regular exercise, and routine checkups.
"""
    elif score < 60:
        return """
Moderate risk.  
Consider reducing sugar intake, increasing physical activity,
and monitoring blood glucose regularly.
"""
    else:
        return """
High risk.  
It would be advisable to consult a doctor,
monitor glucose frequently, and follow a structured diet plan.
"""

st.subheader("Clinical Guidance")
st.write(basic_advice(risk))


# Optional AI Enhancement  #
def get_ai_response(prompt, key):
    import requests

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {key.strip()}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.request(
            "POST",
            url,
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            return f"Status {response.status_code}: {response.text}"

        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error: {str(e)}"



if api_key:
    with st.spinner("Generating AI explanation..."):
        prompt = f"The patient's predicted diabetes risk is {risk:.2f}%. Give practical lifestyle advice."
        ai_text = get_ai_response(prompt, api_key)

        if ai_text:
            st.subheader("AI Based Suggestions")
            st.write(ai_text)
        else:
            st.warning("AI service not responding. Showing standard guidance only.")

#  Correlation Section  #

st.subheader("Feature Correlation Overview")

numeric_df = df_model.select_dtypes(include=np.number)

if target_col in numeric_df.columns:
    corr_values = numeric_df.corr()[target_col].sort_values(ascending=False)
    st.dataframe(corr_values)
else:
    st.write("Correlation data unavailable.")

