
import pandas as pd
import numpy as np

from tabpfn import TabPFNRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load cleaned data
df = pd.read_csv("cleaned_flights.csv")

# Use the same chronological split
df = df.sort_values(
    ["MONTH", "DAY", "SCHEDULED_DEPARTURE"]
).reset_index(drop=True)

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

train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

# Small samples for initial CPU test
train = df.iloc[:train_end].sample(
    n=1000, random_state=42
)

val = df.iloc[train_end:val_end].sample(
    n=100, random_state=42
)

X_train = train[features].copy()
y_train = train["ARRIVAL_DELAY"]

X_val = val[features].copy()
y_val = val["ARRIVAL_DELAY"]

# Encode categorical variables
encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train[categorical_features] = encoder.fit_transform(
    X_train[categorical_features]
)

X_val[categorical_features] = encoder.transform(
    X_val[categorical_features]
)

X_train = X_train.astype(float)
X_val = X_val.astype(float)

print("Initializing TabPFN...", flush=True)

model = TabPFNRegressor(
    device="cpu",
    n_estimators=1
)

print("Fitting TabPFN...", flush=True)

model.fit(X_train, y_train)

print("Making predictions...", flush=True)

predictions = model.predict(X_val)

mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))
r2 = r2_score(y_val, predictions)

print("\nTABPFN INITIAL TEST")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

print("\nTabPFN test complete!")

# Compare all models on the same validation flights
import joblib

results = []

# Mean baseline
training_mean = df.iloc[:train_end]["ARRIVAL_DELAY"].mean()

baseline = np.full(len(y_val), training_mean)

results.append({
    "Model": "Mean Baseline",
    "MAE": mean_absolute_error(y_val, baseline),
    "RMSE": np.sqrt(mean_squared_error(y_val, baseline)),
    "R2": r2_score(y_val, baseline)
})

# Load previously trained models
saved_models = {
    "Linear Regression": "models/linear_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "Gradient Boosting": "models/gradient_boosting.pkl",
    "Deep Neural Network": "models/deep_neural_network.pkl"
}

for name, path in saved_models.items():

    saved_model = joblib.load(path)

    model_predictions = saved_model.predict(
        val[features]
    )

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(
            y_val, model_predictions
        ),
        "RMSE": np.sqrt(mean_squared_error(
            y_val, model_predictions
        )),
        "R2": r2_score(
            y_val, model_predictions
        )
    })

# Add TabPFN results
results.append({
    "Model": "TabPFN",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
})

# Save comparison
comparison = pd.DataFrame(results)
comparison = comparison.sort_values("MAE")

comparison.to_csv(
    "model_comparison.csv",
    index=False
)

# Save TabPFN training reference data
train[features + ["ARRIVAL_DELAY"]].to_csv(
    "tabpfn_training_data.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

print("\nAll five models evaluated successfully!")
