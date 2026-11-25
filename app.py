
import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="FlightIQ Analytics",
    page_icon="✈️",
    layout="wide"
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_flights.csv")

df = load_data()

# Sidebar
st.sidebar.title("✈️ FlightIQ")
st.sidebar.markdown("Flight Delay Prediction & Analytics")
st.sidebar.divider()
st.sidebar.info("Machine Learning Final Project")

# Main title
st.title("✈️ FlightIQ Analytics")
st.subheader("Predict smarter. Travel better.")

st.markdown("""
FlightIQ is a machine learning application designed to
predict flight arrival delays, identify the factors that
influence airline performance, and support smarter
travel and operational decisions.
""")

st.divider()

# Key performance indicators
st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Flights",
    f"{len(df):,}"
)

col2.metric(
    "Average Arrival Delay",
    f"{df['ARRIVAL_DELAY'].mean():.2f} min"
)

col3.metric(
    "Delayed Flights",
    f"{df['IS_DELAYED'].mean() * 100:.2f}%"
)

col4.metric(
    "Airlines",
    df["AIRLINE"].nunique()
)

st.divider()

# Business case
st.header("🎯 Business Case")

st.markdown("""
### The Problem

Flight delays are a major challenge for the aviation
industry. They can lead to missed connections,
increased operational costs, passenger dissatisfaction,
and inefficient airport resource allocation.

### Our Objective

Our objective is to develop a machine learning
application that predicts flight arrival delays using
information available before departure.

### Target Audience

- Airlines and operational planning teams
- Airport operators
- Passengers and travel planners
- Aviation analytics companies

### Research Question

Can machine learning accurately predict flight arrival
delays using scheduled flight information, airline,
airport, distance, and time-related variables?
""")

st.divider()

# Dataset description
st.header("📁 Dataset Presentation")

st.markdown("""
Our dataset comes from the U.S. Department of
Transportation's 2015 Flight Delays and Cancellations
dataset.

The original dataset contains approximately 5.8 million
domestic flight records. We selected a systematic
sample of 50,164 flights to improve computational
efficiency.

After data cleaning and feature engineering, our
dataset contains 45,095 flight records.
""")

col1, col2, col3 = st.columns(3)

col1.metric("Features and Target", df.shape[1])
col2.metric("Departure Airports", df["ORIGIN_AIRPORT"].nunique())
col3.metric("Year", "2015")

st.subheader("Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# Variable descriptions
st.subheader("Key Variables")

variables = pd.DataFrame({
    "Variable": [
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "DISTANCE",
        "SCHEDULED_TIME",
        "DEPARTURE_HOUR",
        "MONTH",
        "DAY_OF_WEEK",
        "ARRIVAL_DELAY"
    ],
    "Description": [
        "Airline identifier",
        "Departure airport",
        "Arrival airport",
        "Flight distance in miles",
        "Scheduled flight duration in minutes",
        "Scheduled departure hour",
        "Month of the flight",
        "Day of the week",
        "Actual arrival delay in minutes"
    ]
})

st.dataframe(variables, hide_index=True,
             use_container_width=True)

st.success("""
Prediction Target: ARRIVAL_DELAY

Our models will predict the number of minutes a flight
arrives early or late relative to its scheduled arrival.
""")

st.divider()

# Data preparation
st.header("🧹 Data Preparation")

st.markdown("""
The following preprocessing steps were performed:

- Removed cancelled and diverted flights.
- Removed records with missing relevant values.
- Retained standard three-letter airport identifiers.
- Created departure and arrival hour variables.
- Created a weekend indicator.
- Defined significant delays as 15 minutes or more.
- Excluded post-departure information from prediction
  features to prevent data leakage.
""")

st.warning("""
Dataset limitation: The data represents historical
U.S. domestic flights from 2015. October records
were excluded because of inconsistent airport
identifiers. Predictions are a historical proof of
concept, not real-time flight forecasts.
""")

st.divider()

# Machine learning overview
st.header("🤖 Machine Learning Approach")

st.markdown("""
We will train and compare five regression models:

1. Linear Regression
2. Random Forest
3. Gradient Boosting
4. Deep Neural Network
5. TabPFN (Tabular Foundation Model)

Model performance will be evaluated using R²,
Mean Absolute Error (MAE), and Root Mean Squared
Error (RMSE).
""")

st.markdown("""
**Dataset source:** [Kaggle - Flight Delays and
Cancellations](https://www.kaggle.com/datasets/usdot/flight-delays)
""")

st.caption("FlightIQ | Machine Learning Final Project")
