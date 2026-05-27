import streamlit as st
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from tensorflow import keras

# --- CONFIGURATION ---
BASE_MODEL_DIR = os.path.join(os.getcwd(), "all_models")

@st.cache_resource
def setup_models():
    # Load your models from the local directories after download
    yolo_face = YOLO(os.path.join(BASE_MODEL_DIR, "yolo/yolov8n.pt"))
    # Load your yolo11x nationality model
    nat_model = YOLO(os.path.join(BASE_MODEL_DIR, "nationality/Race-CLS-FairFace_yolo11x.pt"))
    emotion_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "emotion"))
    age_model = keras.models.load_model(os.path.join(BASE_MODEL_DIR, "age/best_model.h5"), compile=False)
    return yolo_face, nat_model, emotion_pipe, age_model

# --- MAPPING LOGIC ---
def get_mapped_nationality(raw_label):
    # Adjust these keys based on what your yolo11x model outputs exactly
    mapping = {
        "White": "American",
        "Indian": "Indian",
        "Black": "African"
    }
    return mapping.get(raw_label, "Others")

# --- MAIN APP ---
st.title("🌍 Nationality & Attribute Identification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    yolo_face, nat_model, emo_pipe, age_model = setup_models()
    image = Image.open(uploaded_file)
    
    # Pre-processing: If YOLO doesn't detect a face, use the whole image as the face
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = yolo_face(cv_img, classes=[0], conf=0.1, verbose=False)
    
    if results[0].boxes:
        box = results[0].boxes[0] # Take the first detected face
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_crop = image.crop((x1, y1, x2, y2))
    else:
        face_crop = image # Fallback: use full image

    # 1. Predict Nationality
    nat_res = nat_model.predict(face_crop, verbose=False)
    raw_nat = nat_res[0].names[nat_res[0].probs.top1]
    nationality = get_mapped_nationality(raw_nat)
    
    # 2. Predict Emotion (Required for everyone)
    emotion = max(emo_pipe(face_crop), key=lambda x: x['score'])['label']
    
    # 3. Conditional Logic for other attributes
    results_dict = {"Nationality": nationality, "Emotion": emotion}
    
    if nationality == "Indian":
        results_dict["Age"] = "Predicting..." # Call age_model here
        results_dict["Dress Colour"] = "Predicting..." # Add your color logic here
    elif nationality == "American":
        results_dict["Age"] = "Predicting..."
    elif nationality == "African":
        results_dict["Dress Colour"] = "Predicting..."
        
    st.image(image, caption="Input Image")
    st.table(pd.DataFrame([results_dict]))
