from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("heart_model.pkl")

# Input schema
class HeartData(BaseModel):
    image.jpg
    


@app.post("/predict")
def predict(data: HeartData):
    input_data = np.array([[ 
        data.age, data.sex, data.cp, data.trestbps,
        data.chol, data.fbs, data.restecg,
        data.thalach, data.exang, data.oldpeak,
        data.slope, data.ca, data.thal
    ]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0][1])
    }
