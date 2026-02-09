# AuraPath

AuraPath is a simple AI-based clinical decision support prototype focused on early diabetes risk assessment.
The goal of this project is to show how basic patient data can be used to estimate risk and provide understandable insights for both patients and healthcare learners.

## Project Overview

AuraPath is a lightweight MVP created as part of a hackathon project.
It takes a few common clinical inputs and predicts the likelihood of diabetes using a machine learning model. The results are shown in a clear dashboard so the risk level and important factors can be easily understood.

## Features

* Takes basic patient health inputs
* Predicts diabetes risk using a Logistic Regression model
* Shows risk probability in a visual format
* Identifies the main factors influencing the prediction
* Generates simple AI-based lifestyle suggestions and summary notes

## Tech Stack

* Python
* Streamlit
* Pandas and NumPy
* Scikit-learn
* Plotly
* Groq API (used for generating explanations)

## Running the Project Locally

1. Install the required libraries:

pip install -r requirements.txt


2. Run the Streamlit app:

streamlit run trial_mvp.py


