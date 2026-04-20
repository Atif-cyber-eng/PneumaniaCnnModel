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

# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>

/* ===== Animated Background ===== */
body {
    background: linear-gradient(-45deg, #020617, #0f172a, #020617, #1e293b);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* ===== Main Container ===== */
.main {
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(15px);
    border-radius: 25px;
    padding: 25px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
}

/* ===== Title ===== */
h1 {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #4CAF50, #22c55e, #16a34a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ===== Upload Box ===== */
.stFileUploader {
    border: 2px dashed #22c55e;
    border-radius: 20px;
    padding: 20px;
    background: rgba(255,255,255,0.03);
    transition: 0.3s;
}

.stFileUploader:hover {
    background: rgba(34,197,94,0.08);
    transform: scale(1.02);
}

/* ===== Buttons ===== */
.stButton>button {
    background: linear-gradient(135deg, #22c55e, #4CAF50);
    color: white;
    border-radius: 15px;
    padding: 12px 28px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.07);
    box-shadow: 0px 8px 25px rgba(34,197,94,0.7);
}

/* ===== Result Box ===== */
.result-box {
    padding: 30px;
    border-radius: 25px;
    background: rgba(30, 41, 59, 0.8);
    backdrop-filter: blur(12px);
    box-shadow: 0px 15px 40px rgba(0,0,0,0.7);
    animation: fadeInUp 0.7s ease-in-out;
}

/* ===== Animations ===== */
@keyframes fadeInUp {
    from {opacity: 0; transform: translateY(30px);}
    to {opacity: 1; transform: translateY(0);}
}

/* ===== Progress ===== */
.stProgress > div > div {
    background: linear-gradient(90deg, #22c55e, #4CAF50);
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: #020617;
}

/* ===== Image ===== */
img {
    border-radius: 20px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.7);
}

/* ===== Status Badge ===== */
.status-badge {
    display: inline-block;
    padding: 10px 18px;
    border-radius: 25px;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    box-shadow: 0 5px 20px rgba(34,197,94,0.5);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% {box-shadow: 0 0 0 0 rgba(34,197,94,0.6);}
    70% {box-shadow: 0 0 0 20px rgba(34,197,94,0);}
    100% {box-shadow: 0 0 0 0 rgba(34,197,94,0);}
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("""
<h1>🩺 Pneumonia Detection AI</h1>
<p style='text-align:center; color:lightgray;'>
Upload Chest X-ray & get instant AI-powered diagnosis
</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("AI system for detecting Pneumonia using Chest X-rays.")

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

# ---------------- STATUS ----------------
st.markdown("""
<div style="text-align:center;">
    <span class="status-badge">🚀 AI Model Ready</span>
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
        st.markdown("### 🔍 AI is analyzing...")

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
            <div class="result-box" style="border-left:6px solid red;">
                <h2 style="color:red;">⚠️ Pneumonia Detected</h2>
                <p>Confidence: {confidence:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box" style="border-left:6px solid #22c55e;">
                <h2 style="color:#22c55e;">✅ Normal</h2>
                <p>Confidence: {confidence:.2f}</p>
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
