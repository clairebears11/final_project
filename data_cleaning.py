
import pandas as pd
import numpy as np

# Load datasets
flights = pd.read_csv("flights_sample.csv", low_memory=False)
airlines = pd.read_csv("airlines.csv")
airports = pd.read_csv("airports.csv")

print("Original dataset:", flights.shape)

# Remove cancelled and diverted flights
df = flights[
    (flights["CANCELLED"] == 0) &
    (flights["DIVERTED"] == 0)
].copy()

# Select relevant variables
columns = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL",
    "SCHEDULED_TIME",
    "DISTANCE",
    "ARRIVAL_DELAY"
]

df = df[columns].dropna()

# Keep standard airport codes
df = df[
    df["ORIGIN_AIRPORT"].astype(str).str.fullmatch(r"[A-Z]{3}") &
    df["DESTINATION_AIRPORT"].astype(str).str.fullmatch(r"[A-Z]{3}")
].copy()

# Feature engineering
df["DEPARTURE_HOUR"] = (df["SCHEDULED_DEPARTURE"] // 100) % 24
df["ARRIVAL_HOUR"] = (df["SCHEDULED_ARRIVAL"] // 100) % 24
df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([6, 7]).astype(int)
df["IS_DELAYED"] = (df["ARRIVAL_DELAY"] >= 15).astype(int)

df = df.reset_index(drop=True)

# Save cleaned dataset
df.to_csv("cleaned_flights.csv", index=False)

# Display statistics
print("Cleaned dataset:", df.shape)
print("Average delay:", df["ARRIVAL_DELAY"].mean())
print("Delayed flights (%):", df["IS_DELAYED"].mean() * 100)
print("Airlines:", df["AIRLINE"].nunique())
print("Airports:", df["ORIGIN_AIRPORT"].nunique())

print("Data cleaning complete!")
