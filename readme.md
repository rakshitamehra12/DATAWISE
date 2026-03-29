
# DataWise

DataWise is a simple AI-powered Clinical Decision Support (CDS) prototype focused on early diabetes risk prediction.
The project demonstrates how basic patient health data can be analyzed using machine learning to estimate diabetes risk and provide understandable insights.



## Project Overview

DataWise was developed as a hackathon MVP to showcase how data-driven healthcare tools can assist in early disease risk assessment.

The application collects common clinical inputs such as glucose level, BMI, age, blood pressure, etc., and predicts the probability of diabetes using a trained Logistic Regression model.

The results are displayed in a clear and interactive dashboard to help users:

* Understand their predicted risk level
* View probability scores visually
* See important contributing health factors
* Receive simple AI-generated lifestyle suggestions


## Features

* User-friendly health input interface
* Diabetes risk prediction using Logistic Regression
* Probability-based risk scoring
* Interactive data visualization using Plotly
* Identification of key influencing health factors
* AI-generated summary and lifestyle recommendations (via Groq API)



## How It Works

1. User enters clinical values.
2. The trained ML model processes the inputs.
3. The system calculates diabetes risk probability.
4. Visual graphs and insights are displayed.
5. Optional AI integration generates personalized suggestions.

## Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Groq API (for AI-generated explanations)
* Python-dotenv

## Project Structure

DataWise/
│
├── app.py
├── datasetdiabetes.csv
├── README.md
├── requirements.txt
├── .gitignore
├── .env 

## System Architecture

User Input (Streamlit UI)
        ↓
Data Processing Agent
        ↓
ML Prediction Agent (Logistic Regression)
        ↓
Insight Generation Agent
        ↓
AI Recommendation Agent (Groq API)
        ↓
Visualization Layer (Plotly Dashboard)
        ↓
User Output


## Running the Project Locally

### Install required libraries

pip install -r requirements.txt


###  Run the Streamlit app
streamlit run app.py

## API Configuration 

To enable AI-based explanations:

1. Create a `.env` file
2. Add your Groq API key:

```
GROQ_API_KEY=your_api_key_here
```

OR
Enter your API key directly in the Streamlit sidebar interface.

If no API key is provided, the core ML prediction system still works normally.

