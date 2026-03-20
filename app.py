from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json

app = FastAPI()

# 1. The Security Shield (CORS Restriction)
# IMPORTANT: Replace with your actual GitHub Pages URL!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://piyumalsr.github.io/IMG-Classifier/"], 
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# 2. Load the Model 
try:
    with open('config.json', 'r') as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights("model.weights.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# 3. The Public Prediction Endpoint (No API Key Required!)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and format the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((32, 32)) 
        
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0) 
        
        # Make prediction
        prediction = model.predict(img_array)
        
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
