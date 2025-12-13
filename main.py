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

# Input schema
class ChurnInput(BaseModel):
    Age: int
    LoginFrequency: float
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
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)[0]

        return {
            "prediction": int(prediction),
            "churn_probability": float(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
