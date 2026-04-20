import os
import streamlit as st
from PIL import Image
import numpy as np
import time
import gdown

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🩺",
    layout="centered"
)

# ---------------- MODEL ----------------
MODEL_PATH = "pneumonia_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=12XyA6c8ykWGpO5U1eUUCri963BKfsIFg"

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading AI model... Please wait")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

@st.cache_resource
def load_model_cached():
    download_model()
    return load_model(MODEL_PATH)

# ---------------- SPLASH SCREEN ----------------
if "loaded" not in st.session_state:
    st.session_state.loaded = False

if not st.session_state.loaded:
    st.markdown("""
        <div style="text-align:center; padding-top:150px;">
            <h1 style="color:#4CAF50;">🩺 Pneumonia Detection AI</h1>
            <p style="font-size:18px;">Loading AI Model...</p>
        </div>
    """, unsafe_allow_html=True)

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    st.session_state.loaded = True
    st.rerun()

# ---------------- MODERN CSS ----------------
st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg, #0f172a, #020617);
}

/* Main Container */
.main {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 20px;
    color: white;
}

/* Title */
h1 {
    font-weight: 700;
    letter-spacing: 1px;
}

/* Upload Box */
.stFileUploader {
    border: 2px dashed #4CAF50;
    border-radius: 15px;
    padding: 15px;
    background: rgba(255,255,255,0.03);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #4CAF50, #22c55e);
    color: white;
    border-radius: 12px;
    padding: 10px 25px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 5px 20px rgba(76, 175, 80, 0.6);
}

/* Result Card */
.result-box {
    padding: 25px;
    border-radius: 20px;
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(10px);
    box-shadow: 0px 10px 30px rgba(0,0,0,0.6);
    animation: fadeIn 0.6s ease-in-out;
}

/* Animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Progress Bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #22c55e, #4CAF50);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
}

/* Image */
img {
    border-radius: 15px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.6);
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("""
<h1 style='text-align:center; font-size:42px;'>🩺 Pneumonia Detection AI</h1>
<p style='text-align:center; font-size:18px; color:lightgray;'>
Upload Chest X-ray & get instant AI-powered diagnosis
</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    AI system for detecting Pneumonia using Chest X-rays.
    
    ✔ Fast  
    ✔ Accurate  
    ✔ Easy to use  
    """)

    st.header("⚙️ Model Info")
    st.write("Model: CNN")
    st.write("Classes: Normal / Pneumonia")

# ---------------- LOAD MODEL ----------------
try:
    model = load_model_cached()
    st.success("✅ Model Loaded Successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Status Badge
st.markdown("""
<div style="text-align:center; margin-top:10px;">
    <span style="background:#22c55e; padding:8px 15px; border-radius:20px;">
        ✅ AI Model Ready
    </span>
</div>
""", unsafe_allow_html=True)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["jpg", "png", "jpeg"])

# ---------------- PREPROCESS ----------------
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- PREDICTION ----------------
if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

    with col2:
        st.markdown("### 🔍 AI is analyzing the X-ray...")

        with st.spinner("Processing..."):
            img = preprocess(image)
            pred = model.predict(img)

            if pred.shape[-1] == 1:
                prob = pred[0][0]
                label = "Pneumonia" if prob > 0.5 else "Normal"
                confidence = prob if prob > 0.5 else 1 - prob
            else:
                idx = np.argmax(pred)
                labels = ["Normal", "Pneumonia"]
                label = labels[idx]
                confidence = pred[0][idx]

        st.markdown("### 🧾 Result")

        if label == "Pneumonia":
            st.markdown(f"""
            <div class="result-box" style="border-left: 6px solid red;">
                <h2 style="color:red;">⚠️ Pneumonia Detected</h2>
                <p style="font-size:18px;">Confidence: {confidence:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box" style="border-left: 6px solid #22c55e;">
                <h2 style="color:#22c55e;">✅ Normal</h2>
                <p style="font-size:18px;">Confidence: {confidence:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        st.progress(int(confidence * 100))

        with st.expander("🔬 Detailed Output"):
            st.write(pred)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style="text-align:center; font-size:14px;">
Made with ❤️ using Streamlit | AI Medical Project
</p>
""", unsafe_allow_html=True)
