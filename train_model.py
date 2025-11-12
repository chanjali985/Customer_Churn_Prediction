import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Drop customerID
df = df.drop("customerID", axis=1)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Encode categorical columns
for col in df.select_dtypes("object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# ✅ Use only the features you expect in your API request
selected_features = [
    "gender",
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaymentMethod",
]

X = df[selected_features]
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
acc = model.score(X_test, y_test)
print(f"✅ Model trained successfully — Accuracy: {acc:.3f}")

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")
