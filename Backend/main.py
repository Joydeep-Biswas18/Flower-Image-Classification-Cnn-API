from fastapi import FastAPI , File,UploadFile
from typing import Literal,Optional
from pydantic import BaseModel
from model.model import preprocess_image, class_name ,load_model_once
import PIL
import gdown
import numpy as np
import pandas as pd
import io
from PIL import Image


app = FastAPI()

@app.on_event("startup")
def startup_event():
    global model
    model = load_model_once()

@app.get("/")
def home_page():
    return {
        "message":"Hello Welcome MY FlowerVisionAI Model"
    }

@app.post("/predict")
async def predict_img(file: UploadFile =File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    processed_img = preprocess_image(image)
    pred = model.predict(processed_img)

    predicted_class = class_name[np.argmax(pred)]
    confidence = float(np.max(pred))


    pred = pred[0]  # remove batch dimension

    top5_idx = np.argsort(pred)[-5:][::-1]  # top 5 indices (highest first)

    top5_predictions = [{
        "class": class_name[i],
        "confidence": float(pred[i])
    }
    for i in top5_idx
    ]


    return{
        "Flower_class" : predicted_class,
        "Confidence" :confidence,
        "Top Five Prediction" :top5_predictions
    }





