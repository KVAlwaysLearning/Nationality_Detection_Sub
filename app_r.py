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

st.set_page_config(page_title="Nationality & Attribute Analyzer", layout="wide")

# --- PATH CONFIGURATION ---
BASE_MODEL_DIR = "all_models"
# Ensure your GDrive folder has folders named: 'yolo', 'nationality', 'emotion', 'age'
# The 'nationality' folder should contain your 'Race-CLS-FairFace_yolo11x.pt'

@st.cache_resource
def setup_models(drive_folder_id):
    # 1. Download models if they don't exist
    if not os.path.exists(BASE_MODEL_DIR):
        os.makedirs(BASE_MODEL_DIR, exist_ok=True)
        # This is where you pass the ID
        gdown.download_folder(id=drive_folder_id, output=BASE_MODEL_DIR, quiet=False)

    # 2. Load models after download
    yolo_face = YOLO(os.path.join(BASE_MODEL_DIR, "yolo/yolov8n.pt"))
    nat_model = YOLO(os.path.join(BASE_MODEL_DIR, "nationality/Race-CLS-FairFace_yolo11x.pt"))
    emotion_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "emotion"))
    age_model = keras.models.load_model(os.path.join(BASE_MODEL_DIR, "age/best_model.h5"), compile=False)
    
    return yolo_face, nat_model, emotion_pipe, age_model

# --- APP UI ---
st.title("🌍 Nationality & Attribute Identification")

# Define your ID here
DRIVE_ID = "1E2ujnYtnIeIStzI5Eojpg33BurdzShWy"
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    yolo_face, nat_model, emo_pipe, age_model = setup_models(DRIVE_ID)
    
    # Process Image
    image = Image.open(uploaded_file)
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 1. Detect Faces
    results = yolo_face(cv_img, classes=[0], verbose=False)
    
    if results[0].boxes:
        st.image(image, caption="Uploaded Image", use_container_width=True)
        final_results = []
        
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_crop = image.crop((x1, y1, x2, y2))
            
            # Predict Nationality
            nat_res = nat_model.predict(face_crop, verbose=False)
            nationality = nat_res[0].names[nat_res[0].probs.top1]
            
            # Predict Emotion (Required for everyone)
            emotion = max(emo_pipe(face_crop), key=lambda x: x['score'])['label']
            
            entry = {"Nationality": nationality, "Emotion": emotion}
            
            # Conditional Logic
            if nationality == "Indian":
                entry["Age"] = "Predicting..." # Call your age model here
                entry["Dress Colour"] = "Predicting..." # Add your color logic here
            elif nationality == "United States":
                entry["Age"] = "Predicting..."
            elif nationality == "African":
                entry["Dress Colour"] = "Predicting..."
                
            final_results.append(entry)
            
        st.table(pd.DataFrame(final_results))
    else:
        st.warning("No faces detected.")
