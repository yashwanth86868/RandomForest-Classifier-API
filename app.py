
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Create FastAPI application
app = FastAPI(title="Random Forest Prediction API")

# Load trained model
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define input structure
class UserInput(BaseModel):
    age: int
    salary: float

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to Random Forest Prediction API"
    }

# Prediction endpoint
@app.post("/predict")
def predict(data: UserInput):

    input_data = np.array([
        [data.age, data.salary]
    ])

    prediction = model.predict(input_data)

    return {
        "age": data.age,
        "salary": data.salary,
        "prediction": int(prediction[0])
    }
