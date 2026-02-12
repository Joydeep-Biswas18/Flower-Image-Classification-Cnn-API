import tensorflow as tf
import PIL
import numpy as np
import pandas as pd
from PIL import Image
import json
import os
# from .download_model import download_model


# Ensure model exists
# download_model()


# def load_model_once():
#     model = tf.keras.models.load_model("Backend/model/flower_model.keras")
#     return model
# def load_model_once():
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR ,"flower_model.keras")
#     model = tf.keras.models.load_model(model_path,compile=False)
#     return model

# model_path = r"C:\Users\biswa\OneDrive\Documents\New folder\OneDrive\Desktop\ML Projects\Flower-102Image Classification Project\Backend\model\flower_model.keras"

model = None

def load_model_once():
    global model
    if model is None:
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Model loaded successfully")
        except Exception as e:
            print("❌ Model loading failed:", e)
            model = None
    return model
class_file_path = r"C:\Users\biswa\OneDrive\Documents\New folder\OneDrive\Desktop\ML Projects\Flower-102Image Classification Project\Backend\model\class_names.json"
with open(class_file_path,"r") as f:
    class_name = json.load(f)

def preprocess_image(image:Image.Image):
    image = image.resize((224,224))
    image = image.convert("RGB")
    image = np.expand_dims(image,axis=0)
    return image
                    

print("Model path:", model_path)
print("File exists:", os.path.exists(model_path))                    