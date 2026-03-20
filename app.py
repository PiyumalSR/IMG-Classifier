from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Allow your local HTML file and GitHub Pages to talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Model properly 
try:
    with open('config.json', 'r') as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights("model.weights.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((32, 32)) 
        
        # Enforce float32 to keep Keras 3 happy
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0) 
        
        prediction = model.predict(img_array)
        
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
