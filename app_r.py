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

# CONFIGURATION
DRIVE_FOLDER_ID = st.secrets["drive_folder_id"]
BASE_MODEL_DIR = "all_models"

@st.cache_resource
def setup_models():
    # 1. Download Folder
    if not os.path.exists(BASE_MODEL_DIR):
        os.makedirs(BASE_MODEL_DIR, exist_ok=True)
        gdown.download_folder(id=DRIVE_FOLDER_ID, output=BASE_MODEL_DIR, quiet=False)
    
    # DEBUGGER: Print directory structure to find the real path
    st.write("--- Debugging Model Paths ---")
    for root, dirs, files in os.walk(BASE_MODEL_DIR):
        for file in files:
            st.write(f"Found: {os.path.join(root, file)}")
    
    # 2. Define Paths - ADJUST THESE NAMES IF DEBUGGER SHOWS DIFFERENT NAMES
    # Ensure these paths match what the debugger outputs above
    yolo_path = os.path.join(BASE_MODEL_DIR, "yolo/yolov8n.pt")
    nat_path = os.path.join(BASE_MODEL_DIR, "nationality/Race-CLS-FairFace_yolo11x.pt")
    emo_path = os.path.join(BASE_MODEL_DIR, "emotion")
    age_path = os.path.join(BASE_MODEL_DIR, "age/best_model.h5")
    
    # 3. Load Models
    yolo_person = YOLO(yolo_path)
    nat_model = YOLO(nat_path)
    emo_pipe = pipeline("image-classification", model=emo_path)
    age_model = keras.models.load_model(age_path, compile=False)
    
    return yolo_person, nat_model, emo_pipe, age_model

# ... [Keep your existing helper functions: get_mapped_nationality, get_dress_color] ...

# --- MAIN APP ---
st.title("🌍 Nationality & Attribute Identification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        yolo, nat_model, emo_pipe, age_model = setup_models()
        image = Image.open(uploaded_file).convert("RGB")
        
        results = yolo(np.array(image), classes=[0], conf=0.4)
        if results[0].boxes:
            # ... [Keep your existing cropping and prediction logic] ...
            st.success("Analysis Complete")
        else:
            st.error("No person detected.")
    except Exception as e:
        st.error(f"Error during execution: {e}")
