To structure your GitHub repository for the **Nationality & Attribute Detection** project, I have prepared a comprehensive `README.md`. This guide is tailored to explain the project's logic, setup, and repository structure, including the mandatory `requirements.txt` and `packages.txt`.

---

# Nationality & Attribute Detection System

An AI-powered application designed to analyze facial images to predict nationality, emotions, and specific attributes (age, dress color) based on conditional logic.

## 📁 Repository Contents

* **`Nationality_Detection_Working.ipynb`**: The research notebook containing the data preprocessing, model training pipeline, and evaluation metrics (including confusion matrices).
* **`app.py`**: A production-ready Streamlit application that handles image uploads, intelligent object detection, and multi-model inference.
* **`requirements.txt`**: A comprehensive list of Python dependencies required for the project.
* **`packages.txt`**: A list of system-level packages required for specialized support (such as OS-level libraries for image processing).

## 🚀 Features

* **Intelligent Object Detection**: Uses YOLO to localize individuals in an image before analysis.
* **Conditional Attribute Logic**: Implements a rule-based pipeline for output generation:
* **Indians**: Predicts Nationality, Emotion, Age, and Dress Color.
* **Americans**: Predicts Nationality, Emotion, and Age.
* **Africans**: Predicts Nationality, Emotion, and Dress Color.
* **Others**: Predicts Nationality and Emotion.


* **Robust Analytics**: Integrates emotion classification and K-Means clustering for color detection.
* **Intuitive UI**: Provides a clean interface for image uploads and result visualization.

## 🛠️ Setup & Installation

### 1. Prerequisites

Clone this repository:

```bash
git clone https://github.com/KVAlwaysLearning/Nationality_Detection_Sub
cd Nationality_Detection_Sub

```

### 2. Install Dependencies

Install all required libraries and system packages:

```bash
pip install -r requirements.txt
# If deploying on Linux environments (e.g., Streamlit Cloud):
sudo apt-get install -y $(cat packages.txt)

```

**Key Dependencies:**

* `streamlit`: The interactive web interface.
* `transformers` & `ultralytics`: For emotion and object detection inference.
* `tensorflow`/`keras`: For age regression models.
* `webcolors` & `scikit-learn`: For processing color clustering and attributes.
* `gdown`: For automated model weight retrieval from Google Drive.

### 3. Model Initialization

The application automatically downloads pre-trained weights into an `all_models/` directory upon the first launch. Ensure your environment has sufficient permissions to write these files.

## 💻 Usage

### Running the App

Launch the web interface locally:

```bash
streamlit run app.py

```

### Exploring the Research

You can open `Nationality_Detection_Working.ipynb` in any Jupyter-compatible environment to inspect the training methodology, model architecture, and validation results.

## 📂 Project Structure

```text
├── all_models/        # Directory for downloaded model weights or custom trained weights (Refer .ipynb file)
├── app.py             # Streamlit web application
├── Nationality_Detection_Working.ipynb # Research and training notebook
├── requirements.txt   # Python dependencies
├── packages.txt       # System-level dependencies
└── README.md          # Project documentation

```

## 🔗 Links

* **Live App**: [Nationality Detection App](https://nationalitydetectionsub-app.streamlit.app/)
* **GitHub Repo**: [Nationality Detection Repository](https://github.com/KVAlwaysLearning/Nationality_Detection_Sub)

## **Visuals:**

<img width="1392" height="1026" alt="App1" src="https://github.com/user-attachments/assets/3af78be3-2a87-4d51-ba95-0c00219d33c2" />

<img width="944" height="770" alt="App 2" src="https://github.com/user-attachments/assets/d27aec2f-8c08-428a-87d3-547cd25eee0e" />

