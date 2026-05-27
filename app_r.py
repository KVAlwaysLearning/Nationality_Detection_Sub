import streamlit as st
import os
import gdown
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from tensorflow import keras
from sklearn.cluster import KMeans

# 1. Access the Folder ID from Streamlit Secrets
# Define 'drive_folder_id' in your Streamlit Cloud Settings > Secrets
DRIVE_FOLDER_ID = st.secrets["drive_folder_id"]
BASE_MODEL_DIR = "all_models"

@st.cache_resource
def setup_models():
    # 2. Download the entire folder if it doesn't exist locally
    if not os.path.exists(BASE_MODEL_DIR):
        os.makedirs(BASE_MODEL_DIR, exist_ok=True)
        # Downloads the folder recursively
        gdown.download_folder(id=DRIVE_FOLDER_ID, output=BASE_MODEL_DIR, quiet=False)
    
    # 3. Proceed to load your models now that files are in place
    # Example: yolo_model = YOLO(os.path.join(BASE_MODEL_DIR, "yolov8n.pt"))
    return "Models Ready"
    # Load YOLO for Person Detection
    yolo_person = YOLO(os.path.join(BASE_MODEL_DIR, "yolo/yolov8n.pt"))
    # Load Nationality model
    nat_model = YOLO(os.path.join(BASE_MODEL_DIR, "nationality/nat_model_yolo11x.pt"))
    # Load Emotion pipeline
    emo_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "emotion"))
    # Load Age model
    age_model = keras.models.load_model(os.path.join(BASE_MODEL_DIR, "age/best_model.h5"), compile=False)
    return yolo_person, nat_model, emo_pipe, age_model

def get_mapped_nationality(raw_label):
    mapping = {"White": "American", "Indian": "Indian", "Black": "African"}
    return mapping.get(raw_label, "Others")

def get_dress_color(cloth_crop):
    """Predicts dominant color in RGB and returns label."""
    img = np.array(cloth_crop.resize((50, 50))).reshape(-1, 3)
    kmeans = KMeans(n_clusters=1, n_init=5).fit(img)
    rgb = kmeans.cluster_centers_[0].astype(int)
    
    # Simple label mapping
    name = "Other"
    if rgb[0] > 200 and rgb[1] < 100: name = "Red"
    elif rgb[2] > 200: name = "Blue"
    return f"{name} (RGB: {rgb[0]}, {rgb[1]}, {rgb[2]})"

# --- MAIN APP ---
st.title("🌍 Nationality & Attribute Identification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    yolo, nat_model, emo_pipe, age_model = setup_models()
    image = Image.open(uploaded_file).convert("RGB") # Fix for PNG alpha channel
    
    # 1. Detect Person
    results = yolo(np.array(image), classes=[0], conf=0.4)
    if results[0].boxes:
        px1, py1, px2, py2 = map(int, results[0].boxes[0].xyxy[0])
        person_crop = image.crop((px1, py1, px2, py2))
        
        # 2. Extract regions
        face_crop = person_crop.crop((0, 0, person_crop.width, int(person_crop.height * 0.35)))
        cloth_crop = person_crop.crop((0, int(person_crop.height * 0.35), person_crop.width, int(person_crop.height * 0.75)))
        
        # 3. Predictions
        raw_nat = nat_model.predict(face_crop, verbose=False)[0].names[0] # Adjust index based on model output
        nationality = get_mapped_nationality(raw_nat)
        emotion = max(emo_pipe(face_crop), key=lambda x: x['score'])['label']
        
        # Attribute logic
        results_data = {"Nationality": nationality, "Emotion": emotion}
        
        if nationality in ["Indian", "American"]:
            age = int(age_model.predict(np.expand_dims(np.array(face_crop.resize((224, 224)))/255.0, 0), verbose=0)[0][0])
            results_data["Age"] = age
            
        if nationality in ["Indian", "African"]:
            results_data["Dress Colour"] = get_dress_color(cloth_crop)
            
        # 4. Display
        st.image(image.resize((1024, 1024)), caption="Analyzed Image")
        st.table(pd.DataFrame([results_data]))
    else:
        st.error("No person detected.")
