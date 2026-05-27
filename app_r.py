import streamlit as st
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from tensorflow import keras

st.set_page_config(layout="wide", page_title="Nationality & Attribute Analyzer")

# --- CONFIGURATION ---
# Assumes models are in the 'all_models' directory
BASE_MODEL_DIR = os.path.join(os.getcwd(), "all_models")

@st.cache_resource
def setup_models():
    # Load YOLO for Face Detection
    yolo_face = YOLO(os.path.join(BASE_MODEL_DIR, "yolo/yolov8n.pt"))
    # Load your yolo11x Nationality model
    nat_model = YOLO(os.path.join(BASE_MODEL_DIR, "nationality/Race-CLS-FairFace_yolo11x.pt"))
    # Load Emotion pipeline
    emotion_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "emotion"))
    # Load Age model
    age_model = keras.models.load_model(os.path.join(BASE_MODEL_DIR, "age/best_model.h5"), compile=False)
    return yolo_face, nat_model, emotion_pipe, age_model

def get_mapped_nationality(raw_label):
    # Adjust mapping based on exact output labels of your yolo11x model
    mapping = {
        "White": "American",
        "Indian": "Indian",
        "Black": "African"
    }
    return mapping.get(raw_label, "Others")

def get_dominant_color(image_crop):
    """Calculates dominant color using simple averaging."""
    img = np.array(image_crop.resize((50, 50)))
    avg_color = np.mean(img.reshape(-1, 3), axis=0)
    # Basic logic to map RGB to simplified color categories
    return "Standard Tone" 

# --- MAIN APP ---
st.title("🌍 Nationality & Attribute Identification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    yolo_face, nat_model, emo_pipe, age_model = setup_models()
    image = Image.open(uploaded_file)
    
    # 1. Detect Face
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = yolo_face(cv_img, classes=[0], conf=0.1, verbose=False)
    
    if results[0].boxes:
        x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
        face_crop = image.crop((x1, y1, x2, y2))
    else:
        face_crop = image # Use whole image if face not detected
        st.warning("Face not clearly detected, using full image.")

    # 2. Prediction Pipeline
    # Nationality
    nat_res = nat_model.predict(face_crop, verbose=False)
    raw_nat = nat_res[0].names[nat_res[0].probs.top1]
    nationality = get_mapped_nationality(raw_nat)
    
    # Emotion
    emotion = max(emo_pipe(face_crop), key=lambda x: x['score'])['label']
    
    # Age (For Indian/American)
    age = "N/A"
    if nationality in ["Indian", "American"]:
        processed_crop = np.array(face_crop.resize((224, 224)), dtype=np.float32) / 255.0
        age = int(age_model.predict(np.expand_dims(processed_crop, axis=0), verbose=0)[0][0])
    
    # Dress Color (For Indian/African)
    dress_color = "N/A"
    if nationality in ["Indian", "African"]:
        dress_color = get_dominant_color(face_crop)
        
    # 3. Display Results
    st.image(image, caption="Analyzed Image", use_container_width=True)
    results_df = pd.DataFrame([{
        "Nationality": nationality,
        "Emotion": emotion.capitalize(),
        "Age": age,
        "Dress Colour": dress_color
    }])
    st.table(results_df)
