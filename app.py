import os
import streamlit as st
from PIL import Image
import numpy as np
import time

# Try TensorFlow
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)

# ---------------- SPLASH SCREEN ----------------
if "loaded" not in st.session_state:
    st.session_state.loaded = False

if not st.session_state.loaded:
    st.markdown("""
        <div style="text-align:center; padding-top:150px;">
            <h1 style="color:#4CAF50;">🩺 Pneumonia Detection AI</h1>
            <p style="font-size:18px;">Loading model... Please wait</p>
        </div>
    """, unsafe_allow_html=True)

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    st.session_state.loaded = True
    st.rerun()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    color: white;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}
.result-box {
    padding: 20px;
    border-radius: 15px;
    background: #1e293b;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>🩺 Pneumonia Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload Chest X-ray & get instant AI prediction</p>", unsafe_allow_html=True)

MODEL_PATH = "pneumonia_model.h5"

@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This AI model detects Pneumonia from Chest X-ray images using CNN.
    
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
except:
    st.error("❌ Model not found. Place pneumonia_model.h5 file.")
    st.stop()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg", "png", "jpeg"])

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
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.write("### 🔍 Analyzing...")

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

        # ---------------- RESULT UI ----------------
        st.markdown("### 🧾 Result")

        if label == "Pneumonia":
            st.markdown(f"""
            <div class="result-box" style="border-left: 6px solid red;">
                <h2 style="color:red;">⚠️ {label}</h2>
                <p>Confidence: {confidence:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box" style="border-left: 6px solid green;">
                <h2 style="color:lightgreen;">✅ {label}</h2>
                <p>Confidence: {confidence:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        # ---------------- PROGRESS BAR ----------------
        st.progress(int(confidence * 100))

        # ---------------- EXTRA ----------------
        with st.expander("🔬 Detailed Output"):
            st.write(pred)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style="text-align:center; font-size:14px;">
Made with ❤️ using Streamlit | AI Medical Project
</p>
""", unsafe_allow_html=True)
