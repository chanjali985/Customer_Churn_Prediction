from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import traceback

app = FastAPI(title="Customer Churn API")

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print("⚠️ Error loading model:", e)
    traceback.print_exc()

class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: int
    PaymentMethod: int

@app.post("/predict")
def predict(data: CustomerData):
    try:
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)[0]
        return {"churn": bool(prediction)}
    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running 🚀"}
