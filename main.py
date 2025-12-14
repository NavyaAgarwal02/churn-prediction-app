from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Churn Prediction API")

# Load trained model
try:
    model = joblib.load("churn_model.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# Input schema (strictly numeric)
class ChurnInput(BaseModel):
    Age: int
    LoginFrequency: int
    AvgSessionTime: float
    SupportTickets: int
    Gender_M: int
    SubscriptionType_Premium: int
    SubscriptionType_VIP: int

FEATURES = [
    "Age",
    "LoginFrequency",
    "AvgSessionTime",
    "SupportTickets",
    "Gender_M",
    "SubscriptionType_Premium",
    "SubscriptionType_VIP"
]

@app.get("/")
def health_check():
    return {"status": "API running successfully"}

@app.post("/predict")
def predict_churn(data: ChurnInput):
    try:
        # Convert input to DataFrame with correct column order
        input_df = pd.DataFrame([data.dict()])[FEATURES]

        # Make prediction
        churn_pred = model.predict(input_df)[0]

        # Compute probability if available
        if hasattr(model, "predict_proba"):
            churn_prob = model.predict_proba(input_df)[0][1]
        else:
            churn_prob = None

        return {
            "churn_prediction": int(churn_pred),
            "churn_probability": float(churn_prob) if churn_prob is not None else None
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Inference failed: {str(e)}"
        )
