import streamlit as st
import os
import gdown
import pandas as pd
import numpy as np
import webcolors
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from tensorflow import keras
from sklearn.cluster import KMeans

# 1. Page Config
st.set_page_config(page_title="Identity & Attribute ID", layout="wide")

# 2. Configuration & Secrets
DRIVE_FOLDER_ID = st.secrets["drive_folder_id"]
BASE_MODEL_DIR = "all_models"

# 3. Cache Clearing Logic
if 'last_file' not in st.session_state:
    st.session_state.last_file = None

@st.cache_resource
def setup_models():
    if not os.path.exists(BASE_MODEL_DIR):
        os.makedirs(BASE_MODEL_DIR, exist_ok=True)
        gdown.download_folder(id=DRIVE_FOLDER_ID, output=BASE_MODEL_DIR, quiet=False)
    
    yolo_person = YOLO(os.path.join(BASE_MODEL_DIR, "yolo/yolov8n.pt"))
    nat_model = YOLO(os.path.join(BASE_MODEL_DIR, "nationality/nat_model_yolo11x.pt"))
    emo_pipe = pipeline("image-classification", model=os.path.join(BASE_MODEL_DIR, "emotion"))
    age_model = keras.models.load_model(os.path.join(BASE_MODEL_DIR, "age/best_model.h5"), compile=False)
    
    return yolo_person, nat_model, emo_pipe, age_model

def get_mapped_nationality(raw_label):
    mapping = {"White": "Americans", "Indian": "Indians", "Black": "Africans"}
    return mapping.get(raw_label, "Others")

def get_closest_color_name(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_dress_color(cloth_crop):
    img = np.array(cloth_crop.resize((50, 50))).reshape(-1, 3)
    kmeans = KMeans(n_clusters=1, n_init=5).fit(img)
    rgb = tuple(kmeans.cluster_centers_[0].astype(int))
    color_name = get_closest_color_name(rgb)
    return f"{color_name.capitalize()} (RGB: {rgb[0]}, {rgb[1]}, {rgb[2]})"

# 4. Main App
st.title("🌍 Identity & Attribute Identification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    if file_id != st.session_state.last_file:
        st.cache_resource.clear()
        st.session_state.last_file = file_id

    yolo_p, nat_model, emo_pipe, age_model = setup_models()
    image = Image.open(uploaded_file).convert("RGB")
    
    p_results = yolo_p(np.array(image), classes=[0], conf=0.5)
    
    if p_results[0].boxes:
        best_p = max(p_results[0].boxes, key=lambda b: b.conf)
        px1, py1, px2, py2 = map(int, best_p.xyxy[0])
        person_crop = image.crop((px1, py1, px2, py2))
        
        # Analysis
        nat_res = nat_model.predict(person_crop, verbose=False)
        raw_nat = nat_res[0].names[int(nat_res[0].probs.top1)]
        nationality = get_mapped_nationality(raw_nat)
        
        emotion = max(emo_pipe(person_crop), key=lambda x: x['score'])['label']
        age = int(age_model.predict(np.expand_dims(np.array(person_crop.resize((224, 224)))/255.0, 0), verbose=0)[0][0])
        
        cloth_crop = person_crop.crop((0, int(person_crop.height * 0.4), person_crop.width, person_crop.height))
        dress = get_dress_color(cloth_crop)
        
        # Output
        st.image(image.resize((1024, 1024)), caption="Analyzed Image", use_container_width=False, width=1024)
        st.table(pd.DataFrame([{
            "Nationality": nationality, "Age": age, "Emotion": emotion.capitalize(), "Dress": dress
        }]))
    else:
        st.error("No person detected.")
