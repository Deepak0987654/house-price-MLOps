from fastapi import FastAPI, HTTPException
from app.schemas.response import PredictionResponse
from app.schemas.input import UserInput
from app.services.predict import predict_price
from app.core.model_loader import model_version

app = FastAPI()

# human readable
@app.get("/")
def home():
    return {"message": "Housing Price Prediction API is running"}

# machine readable
@app.get('/health')
def health_check():
    return {"status":"ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_sale_price(user_input: UserInput):
    try:
        user_dict = user_input.model_dump()

        result = predict_price(user_dict)
        return PredictionResponse(
            prediction = result["prediction"],
            model_version = model_version,
            timestamp = result["timestamp"]
        )
    
    except Exception as e:

        return HTTPException(status_code=500, content=str(e))