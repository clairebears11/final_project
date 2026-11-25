
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="FlightIQ | Data Visualization",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_flights.csv")
    airlines = pd.read_csv("airlines.csv")
    airports = pd.read_csv("airports.csv")

    airline_names = airlines.set_index(
        "IATA_CODE"
    )["AIRLINE"].to_dict()

    airport_names = airports.set_index(
        "IATA_CODE"
    )["AIRPORT"].to_dict()

    df["AIRLINE_NAME"] = df["AIRLINE"].map(
        airline_names
    ).fillna(df["AIRLINE"])

    df["AIRPORT_NAME"] = df["ORIGIN_AIRPORT"].map(
        airport_names
    ).fillna(df["ORIGIN_AIRPORT"])

    return df

df = load_data()

st.title("📊 Flight Delay Analytics")

st.markdown("""
Explore historical flight performance and identify
patterns in airline, airport, and scheduling data.
""")

# Sidebar filters
st.sidebar.header("🔎 Dashboard Filters")

selected_airlines = st.sidebar.multiselect(
    "Select Airlines",
    sorted(df["AIRLINE_NAME"].unique()),
    default=sorted(df["AIRLINE_NAME"].unique())
)

selected_months = st.sidebar.multiselect(
    "Select Months",
    sorted(df["MONTH"].unique()),
    default=sorted(df["MONTH"].unique())
)

filtered = df[
    df["AIRLINE_NAME"].isin(selected_airlines) &
    df["MONTH"].isin(selected_months)
].copy()

if filtered.empty:
    st.warning("Please select at least one airline and month.")
    st.stop()

# KPI metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Flights", f"{len(filtered):,}")

col2.metric(
    "Average Delay",
    f"{filtered['ARRIVAL_DELAY'].mean():.2f} min"
)

col3.metric(
    "Delayed Flights",
    f"{filtered['IS_DELAYED'].mean()*100:.2f}%"
)

col4.metric(
    "Median Delay",
    f"{filtered['ARRIVAL_DELAY'].median():.2f} min"
)

st.divider()

# CHART 1: Airline comparison
st.header("1. Airline Performance")

airline_stats = filtered.groupby(
    "AIRLINE_NAME"
).agg(
    Average_Delay=("ARRIVAL_DELAY", "mean"),
    Total_Flights=("ARRIVAL_DELAY", "count"),
    Delay_Rate=("IS_DELAYED", "mean")
).reset_index()

airline_stats["Delay_Rate"] *= 100

airline_stats = airline_stats.sort_values(
    "Average_Delay"
)

fig = px.bar(
    airline_stats,
    x="Average_Delay",
    y="AIRLINE_NAME",
    orientation="h",
    color="Average_Delay",
    color_continuous_scale="RdYlGn_r",
    hover_data=["Total_Flights", "Delay_Rate"],
    labels={
        "Average_Delay": "Average Delay (minutes)",
        "AIRLINE_NAME": "Airline"
    },
    title="Average Arrival Delay by Airline"
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
This comparison helps identify airlines with
higher or lower historical average arrival delays.
Differences may also reflect routes, airports,
weather, and scheduling conditions.
""")

st.divider()

# CHART 2: Monthly trends
st.header("2. Seasonal Delay Trends")

monthly = filtered.groupby("MONTH")[
    "ARRIVAL_DELAY"
].mean().reset_index()

fig = px.line(
    monthly,
    x="MONTH",
    y="ARRIVAL_DELAY",
    markers=True,
    labels={
        "MONTH": "Month",
        "ARRIVAL_DELAY": "Average Delay (minutes)"
    },
    title="Average Flight Delay by Month"
)

fig.update_traces(
    line_color="#2563EB",
    line_width=3
)

fig.update_xaxes(dtick=1)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "October is excluded because of inconsistent "
    "airport identifiers in the original dataset."
)

st.divider()

# CHART 3: Departure hour
st.header("3. Best Time to Fly")

hourly = filtered.groupby("DEPARTURE_HOUR")[
    "ARRIVAL_DELAY"
].mean().reset_index()

fig = px.bar(
    hourly,
    x="DEPARTURE_HOUR",
    y="ARRIVAL_DELAY",
    color="ARRIVAL_DELAY",
    color_continuous_scale="RdYlGn_r",
    labels={
        "DEPARTURE_HOUR": "Scheduled Departure Hour",
        "ARRIVAL_DELAY": "Average Delay (minutes)"
    },
    title="Average Arrival Delay by Departure Hour"
)

fig.update_xaxes(dtick=1)

st.plotly_chart(fig, use_container_width=True)

st.info("""
Passengers can use historical scheduling patterns
to compare departure times. These patterns do not
guarantee future flight performance.
""")

st.divider()

# CHART 4: Airport performance
st.header("4. Airport Performance")

airport_stats = filtered.groupby(
    "AIRPORT_NAME"
).agg(
    Average_Delay=("ARRIVAL_DELAY", "mean"),
    Total_Flights=("ARRIVAL_DELAY", "count")
).reset_index()

# Avoid ranking airports with very few flights
airport_stats = airport_stats[
    airport_stats["Total_Flights"] >= 50
]

airport_stats = airport_stats.nlargest(
    10, "Average_Delay"
)

if airport_stats.empty:
    st.info("Not enough flights for airport rankings.")
else:
    fig = px.bar(
        airport_stats.sort_values("Average_Delay"),
        x="Average_Delay",
        y="AIRPORT_NAME",
        orientation="h",
        color="Average_Delay",
        color_continuous_scale="OrRd",
        hover_data=["Total_Flights"],
        labels={
            "Average_Delay": "Average Delay (minutes)",
            "AIRPORT_NAME": "Departure Airport"
        },
        title="10 Airports with Highest Average Delays"
    )

    st.plotly_chart(fig, use_container_width=True)

st.caption("Only airports with at least 50 flights are included.")

st.divider()

# CHART 5: Delay distribution
st.header("5. Arrival Delay Distribution")

fig = px.histogram(
    filtered,
    x="ARRIVAL_DELAY",
    nbins=100,
    color_discrete_sequence=["#2563EB"],
    labels={
        "ARRIVAL_DELAY": "Arrival Delay (minutes)"
    },
    title="Distribution of Arrival Delays"
)

fig.update_xaxes(range=[-60, 180])

fig.add_vline(
    x=15,
    line_dash="dash",
    line_color="red",
    annotation_text="15-minute delay threshold"
)

st.plotly_chart(fig, use_container_width=True)

st.caption("""
The displayed range is limited to -60 to 180 minutes
for readability. Extreme delays remain in the dataset.
Negative values represent early arrivals.
""")

st.divider()

# CHART 6: Delay heatmap
st.header("6. Flight Delay Heatmap")

heatmap = filtered.pivot_table(
    values="ARRIVAL_DELAY",
    index="DAY_OF_WEEK",
    columns="DEPARTURE_HOUR",
    aggfunc="mean"
)

heatmap = heatmap.reindex(
    index=range(1, 8),
    columns=range(24)
)

heatmap.index = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

fig = px.imshow(
    heatmap,
    color_continuous_scale="RdYlGn_r",
    aspect="auto",
    labels={
        "x": "Departure Hour",
        "y": "Day of Week",
        "color": "Average Delay"
    },
    title="Average Delay by Day and Departure Hour"
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
The heatmap identifies combinations of departure
hours and days associated with higher historical
arrival delays. Red indicates higher average delays,
while green indicates lower average delays.
""")

st.divider()

# Extra feature: Download filtered data
st.header("📥 Download Filtered Data")

st.download_button(
    label="Download CSV",
    data=filtered.to_csv(index=False),
    file_name="flightiq_filtered_data.csv",
    mime="text/csv"
)

st.caption("FlightIQ | Historical Flight Analytics")
