from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Churn Prediction API")

# Load trained model
try:
    model = joblib.load("churn_model.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# Input schema (STRICTLY numeric)
class ChurnInput(BaseModel):
    Age: int
    LoginFrequency: int
    AvgSessionTime: float
    SupportTickets: int
    Gender_M: int
    SubscriptionType_Premium: int
    SubscriptionType_VIP: int

@app.get("/")
def health_check():
    return {"status": "API running successfully"}

@app.post("/predict")
def predict_churn(data: ChurnInput):
    try:
        # Convert input to DataFrame with correct column order
        input_df = pd.DataFrame([{
            "Age": data.Age,
            "LoginFrequency": data.LoginFrequency,
            "AvgSessionTime": data.AvgSessionTime,
            "SupportTickets": data.SupportTickets,
            "Gender_M": data.Gender_M,
            "SubscriptionType_Premium": data.SubscriptionType_Premium,
            "SubscriptionType_VIP": data.SubscriptionType_VIP
        }])

        # Prediction
        churn_pred = model.predict(input_df)[0]

        # Probability (if supported)
        if hasattr(model, "predict_proba"):
            churn_prob = model.predict_proba(input_df)[0][1]
        else:
            churn_prob = None

        return {
            "churn_prediction": int(churn_pred),
            "churn_probability": churn_prob
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Inference failed: {str(e)}"
        )
