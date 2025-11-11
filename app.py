from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn API")
model = joblib.load("model.pkl")

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
