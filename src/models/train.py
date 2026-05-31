import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.pipelines.pipeline import create_pipeline  # your pipeline

import joblib
from sklearn.base import clone
from pathlib import Path
import os

# Load data
df = pd.read_csv("data/raw/train.csv")

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Detect features
num_features = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_features = X_train.select_dtypes(include=["object"]).columns

# Create pipeline
pipeline = create_pipeline(num_features, cat_features)

# Models to compare
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=100),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, verbosity=0),
    "LightGBM": LGBMRegressor(n_estimators=100)
}

mlflow.set_experiment("house-price-prediction")

best_rmse = float("inf")
best_pipeline = None
best_model_name = None

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        current_pipeline = clone(pipeline)

        # Replace model in pipeline
        current_pipeline.set_params(model__regressor=model)

        # Train
        current_pipeline.fit(X_train, y_train)

        # Predict
        preds = current_pipeline.predict(X_test)

        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log
        mlflow.log_param("model", name)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(current_pipeline, "model")

        print(f"{name} RMSE: {rmse}")

        # BEST MODEL LOGIC
        if rmse < best_rmse:
            best_rmse = rmse
            best_pipeline = current_pipeline
            best_model_name = name

print(f"\n✅ Best Model: {best_model_name} with RMSE: {best_rmse}")
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # go to project root
ARTIFACTS_DIR = BASE_DIR / "artifacts"

ARTIFACTS_DIR.mkdir(exist_ok=True)

joblib.dump(best_pipeline, ARTIFACTS_DIR / "best_model.pkl")

print("Saved at:", ARTIFACTS_DIR / "best_model.pkl")