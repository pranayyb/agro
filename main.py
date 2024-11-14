# To run this app : "uvicorn main:app --reload"

import os
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import pickle
import numpy as np
from inference_sdk import InferenceHTTPClient
import io
from PIL import Image

with open("models/crop_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

with open("models/label_encoder.pkl", "rb") as encoder_file:
    loaded_label_encoder = pickle.load(encoder_file)

app = FastAPI()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", api_key="yjGS9C041cEmorXMHwZI"
)

plant_diseases_to_fertilizers = {
    "Apple Scab Leaf": {
        "fertilizer": "Nitrogen-rich fertilizers",
        "treatment": "Fungicides containing captan or sulfur",
    },
    "Apple leaf": {
        "fertilizer": "Balanced NPK fertilizers (10-10-10)",
        "treatment": "Fungicides such as myclobutanil or mancozeb",
    },
    "Apple rust leaf": {
        "fertilizer": "Phosphorus-rich fertilizers",
        "treatment": "Fungicides containing triadimefon or sulfur",
    },
    "Bell_pepper leaf spot": {
        "fertilizer": "Balanced NPK fertilizer (10-10-10)",
        "treatment": "Copper-based fungicides or mancozeb",
    },
    "Bell_pepper leaf": {
        "fertilizer": "Nitrogen-rich fertilizers",
        "treatment": "Fungicides such as chlorothalonil",
    },
    "Blueberry leaf": {
        "fertilizer": "Acidic fertilizers (Ammonium sulfate)",
        "treatment": "Fungicides containing sulfur or copper",
    },
    "Cherry leaf": {
        "fertilizer": "Balanced fertilizer with a focus on potassium",
        "treatment": "Fungicides containing copper or neem oil",
    },
    "Corn Gray leaf spot": {
        "fertilizer": "High-nitrogen fertilizer",
        "treatment": "Fungicides containing strobilurin or chlorothalonil",
    },
    "Corn leaf blight": {
        "fertilizer": "Phosphorus-rich fertilizers",
        "treatment": "Fungicides with azoxystrobin or propiconazole",
    },
    "Corn rust leaf": {
        "fertilizer": "High-nitrogen fertilizers",
        "treatment": "Fungicides containing triazole or chlorothalonil",
    },
    "Peach leaf": {
        "fertilizer": "Balanced NPK fertilizer with a focus on potassium",
        "treatment": "Fungicides containing copper or sulfur",
    },
    "Potato leaf early blight": {
        "fertilizer": "Balanced NPK fertilizers, especially with potassium",
        "treatment": "Fungicides such as chlorothalonil or mancozeb",
    },
    "Potato leaf late blight": {
        "fertilizer": "Phosphorus-rich fertilizers",
        "treatment": "Fungicides containing metalaxyl or mefenoxam",
    },
    "Potato leaf": {
        "fertilizer": "Balanced fertilizers (NPK 10-10-10)",
        "treatment": "Fungicides such as chlorothalonil or copper sulfate",
    },
    "Raspberry leaf": {
        "fertilizer": "Balanced NPK fertilizer (10-10-10)",
        "treatment": "Fungicides containing sulfur or copper",
    },
    "Soyabean leaf": {
        "fertilizer": "Balanced NPK fertilizer (15-15-15)",
        "treatment": "Fungicides containing azoxystrobin or chlorothalonil",
    },
    "Squash Powdery mildew leaf": {
        "fertilizer": "High-potassium fertilizers",
        "treatment": "Fungicides such as sulfur or potassium bicarbonate",
    },
    "Strawberry leaf": {
        "fertilizer": "Balanced NPK fertilizer with an emphasis on potassium",
        "treatment": "Fungicides containing chlorothalonil or sulfur",
    },
    "Tomato Early blight leaf": {
        "fertilizer": "Balanced NPK fertilizer with high potassium",
        "treatment": "Fungicides like chlorothalonil or copper-based treatments",
    },
    "Tomato Septoria leaf spot": {
        "fertilizer": "Balanced NPK fertilizers (10-10-10)",
        "treatment": "Fungicides containing chlorothalonil or mancozeb",
    },
    "Tomato leaf bacterial spot": {
        "fertilizer": "Balanced NPK fertilizer with a focus on potassium",
        "treatment": "Copper-based fungicides",
    },
    "Tomato leaf late blight": {
        "fertilizer": "Phosphorus-rich fertilizers",
        "treatment": "Fungicides containing mefenoxam or copper",
    },
    "Tomato leaf mosaic virus": {
        "fertilizer": "Nutrient-rich fertilizers to improve plant vigor",
        "treatment": "No chemical treatment, remove infected plants",
    },
    "Tomato leaf yellow virus": {
        "fertilizer": "Balanced NPK fertilizers",
        "treatment": "No chemical treatment, remove infected plants",
    },
    "Tomato leaf": {
        "fertilizer": "Balanced NPK fertilizer (10-10-10)",
        "treatment": "Fungicides containing chlorothalonil or copper",
    },
    "Tomato mold leaf": {
        "fertilizer": "High-potassium fertilizers",
        "treatment": "Fungicides like potassium bicarbonate or sulfur",
    },
    "Tomato two spotted spider mites leaf": {
        "fertilizer": "Balanced NPK fertilizer",
        "treatment": "Miticides or insecticidal soaps",
    },
    "grape leaf black rot": {
        "fertilizer": "Balanced NPK fertilizers with potassium",
        "treatment": "Fungicides like sulfur or copper-based treatments",
    },
    "grape leaf": {
        "fertilizer": "Balanced NPK fertilizers (10-10-10)",
        "treatment": "Fungicides such as copper sulfate",
    },
}


class CropPredictionInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.post("/predict-crop")
def predict_crop(data: CropPredictionInput):
    try:
        X_testing = np.array(
            [
                data.nitrogen,
                data.phosphorus,
                data.potassium,
                data.temperature,
                data.humidity,
                data.ph,
                data.rainfall,
            ]
        ).reshape(1, -1)

        X_testing_scaled = loaded_scaler.transform(X_testing)

        y_testing = model.predict(X_testing_scaled)[0]

        label = loaded_label_encoder.inverse_transform([y_testing])[0]

        return {"predicted_crop": label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    try:
        temp_file_path = f"temp_{file.filename}"

        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        result = CLIENT.infer(temp_file_path, model_id="plants-final/1")

        predicted_class = result["predictions"][0]["class"]
        confidence = result["predictions"][0]["confidence"]

        os.remove(temp_file_path)

        disease_info = plant_diseases_to_fertilizers.get(predicted_class, None)

        if disease_info:
            return {
                "predicted_class": predicted_class,
                "confidence": f"{confidence * 100:.2f}%",
                "fertilizer_recommendation": disease_info["fertilizer"],
                "treatment_recommendation": disease_info["treatment"],
            }
        else:
            return {
                "predicted_class": predicted_class,
                "confidence": f"{confidence * 100:.2f}%",
                "fertilizer_recommendation": "No specific fertilizer recommendation",
                "treatment_recommendation": "No specific treatment recommendation",
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
