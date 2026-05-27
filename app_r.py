import streamlit as st
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from tensorflow import keras
from sklearn.cluster import KMeans

# --- CONFIGURATION ---
BASE_MODEL_DIR = "all_models"

@st.cache_resource
def setup_models():
    yolo_person = YOLO(os.path.join(BASE_MODEL_DIR, "yolo/yolov8n.pt"))
    nat_model = YOLO(os.path.join(BASE_MODEL_DIR, "nationality/Race-CLS-FairFace_yolo11x.pt"))
    emo_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "emotion"))
    age_model = keras.models.load_model(os.path.join(BASE_MODEL_DIR, "age/best_model.h5"), compile=False)
    return yolo_person, nat_model, emo_pipe, age_model

def get_dominant_color_label(clothing_crop):
    img = np.array(clothing_crop.resize((100, 100)))
    pixels = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, n_init=5).fit(pixels)
    
    # Get dominant color
    dominant_rgb = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    
    # Mapping
    colors = {"Red": [255,0,0], "Blue": [0,0,255], "Black": [0,0,0], "White": [255,255,255]}
    label = min(colors.keys(), key=lambda c: np.linalg.norm(np.array(colors[c]) - dominant_rgb))
    return f"{label} (RGB: {int(dominant_rgb[0])}, {int(dominant_rgb[1])}, {int(dominant_rgb[2])})"

# --- APP ---
st.title("🌍 Nationality & Attribute Identification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    yolo, nat_model, emo_pipe, age_model = setup_models()
    image = Image.open(uploaded_file)

    # 1. Convert to RGB to remove the Alpha channel (transparency)
    image_rgb = image.convert("RGB")
    
    # 2. Convert to numpy array
    img_array = np.array(image_rgb)
        
        
    # 1. Detect Person
    results = yolo(np.array(image), classes=[0], conf=0.4)
    
    if results[0].boxes:
        px1, py1, px2, py2 = map(int, results[0].boxes[0].xyxy[0])
        person_crop = image.crop((px1, py1, px2, py2))
        
        # 2. Define Regions (Relative to Person)
        # Face is roughly top 30% of the person box
        face_crop = person_crop.crop((0, 0, person_crop.width, int(person_crop.height * 0.35)))
        # Clothing is roughly middle 40% of the person box
        cloth_crop = person_crop.crop((0, int(person_crop.height * 0.35), person_crop.width, int(person_crop.height * 0.75)))
        
        # 3. Predict Attributes
        nat_res = nat_model.predict(face_crop, verbose=False)
        nationality = nat_res[0].names[nat_res[0].probs.top1]
        
        emotion = max(emo_pipe(face_crop), key=lambda x: x['score'])['label']
        
        age = int(age_model.predict(np.expand_dims(np.array(face_crop.resize((224, 224)))/255.0, 0), verbose=0)[0][0])
        
        dress = get_dominant_color_label(cloth_crop)
        
        # 4. Display
        st.image(image.resize((1024, 1024)), caption="Analyzed Image", use_container_width=True)
        st.table(pd.DataFrame([{
            "Nationality": nationality, "Age": age, "Emotion": emotion, "Dress": dress
        }]))
    else:
        st.error("No person detected.")
