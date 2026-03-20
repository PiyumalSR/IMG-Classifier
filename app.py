from fastapi import FastAPI, Security, HTTPException, File, UploadFile, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json

app = FastAPI()

# 1. Enable CORS (Allows your web UI to talk to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. API Key Security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    return api_key_header 

# 3. Load the Model (Architecture + Weights)
try:
    with open('config.json', 'r') as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights("model.weights.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# 4. The Image Prediction Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    try:
        # Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize to 32x32 based on your config.json!
        image = image.resize((32, 32)) 
        
        # Convert image to numpy array and normalize (0 to 1)
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0) 
        
        # Make prediction
        prediction = model.predict(img_array)
        
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))