import pandas as pd
from datetime import datetime, timezone
from app.core.model_loader import model, features, defaults

def build_full_input(user_input: dict):
    data = defaults.copy()
    data.update(user_input)
    return data

def prepare_df(data: dict):
    df = pd.DataFrame([data])
    return df[features]


def predict_price(user_input: dict):
    full_input = build_full_input(user_input)
    df = prepare_df(full_input)

    prediction = float(model.predict(df)[0])
    return {
        "prediction": prediction,
        "timestamp": datetime.now(timezone.utc)
    }