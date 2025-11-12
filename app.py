from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Customer Churn API")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

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
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"churn": bool(prediction)}

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running 🚀"}

