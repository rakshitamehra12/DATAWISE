# Aura-Path
An AI-driven diabetes clinical decision support system with risk prediction, explainability, and patient-friendly insights.
# AuraPath (MVP)
AuraPath is a lightweight clinical decision support prototype built for a hackathon.
It estimates diabetes risk using basic patient data and presents the results through
an interactive dashboard.

## What the app does
- Collects patient clinical inputs
- Predicts diabetes risk using a logistic regression model
- Displays risk level and probability visually
- Highlights key risk-driving factors
- Generates AI-based lifestyle guidance and clinical summaries

## Tech stack
- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Plotly
- Groq API (for AI-generated explanations)

## How to run locally
```bash
pip install -r requirements.txt
streamlit run trial_mvp.py

