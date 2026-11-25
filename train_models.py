from sklearn.neural_network import MLPRegressor

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load cleaned dataset
df = pd.read_csv("cleaned_flights.csv")

# Sort chronologically
df = df.sort_values(
    ["MONTH", "DAY", "SCHEDULED_DEPARTURE"]
).reset_index(drop=True)

# Select prediction features
categorical_features = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT"
]

numerical_features = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "DEPARTURE_HOUR",
    "ARRIVAL_HOUR",
    "SCHEDULED_TIME",
    "DISTANCE",
    "IS_WEEKEND"
]

features = categorical_features + numerical_features

X = df[features]
y = df["ARRIVAL_DELAY"]

# Split data into training, validation and testing
train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)

# Define machine learning models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        random_state=42,
        n_jobs=2
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "Deep Neural Network": MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        random_state=42
    )
}

# Create folder for trained models
os.makedirs("models", exist_ok=True)

results = []

# Simple baseline: predict the training average
baseline = np.full(len(y_val), y_train.mean())

results.append({
    "Model": "Mean Baseline",
    "MAE": mean_absolute_error(y_val, baseline),
    "RMSE": np.sqrt(
        mean_squared_error(y_val, baseline)
    ),
    "R2": r2_score(y_val, baseline)
})

# Train and evaluate each model
for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )

    r2 = r2_score(y_val, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    filename = name.lower().replace(" ", "_")

    joblib.dump(
        pipeline,
        f"models/{filename}.pkl"
    )

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")

# Save model performance
results_df = pd.DataFrame(results)

results_df = results_df.sort_values("MAE")

results_df.to_csv(
    "model_results.csv",
    index=False
)

print("\nMODEL PERFORMANCE")
print(results_df.to_string(index=False))

print("\nModel training complete!")
