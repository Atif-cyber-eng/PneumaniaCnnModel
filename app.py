import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps
import io

st.set_page_config(page_title="Pneumonia X-ray Detector", layout="centered")

st.title("Pneumonia Detector — Chest X-ray")
st.markdown("Upload a chest X-ray image (PA/AP). The model predicts NORMAL vs PNEUMONIA.")

# Path to model in repo; if using Drive or remote, give URL or local path after download
MODEL_PATH = "pneumonia_model.h5"  # ensure this is in repo or same folder

@st.cache_resource(show_spinner=False)
def load_my_model(path):
    model = load_model(path)
    return model

model = load_my_model(MODEL_PATH)

def preprocess_image(img: Image.Image, target_size=(224,224)):
    # convert to RGB, resize, scale
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = ImageOps.fit(img, target_size, Image.ANTIALIAS)
    arr = np.asarray(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

uploaded_file = st.file_uploader("Upload X-ray image (jpg/png)", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded image", use_column_width=True)
    st.write("")
    if st.button("Predict"):
        X = preprocess_image(img, target_size=(224,224))
        pred = model.predict(X)[0][0]
        # model outputs probability for class 1 (PNEUMONIA) if trained that way
        prob = float(pred)
        if prob >= 0.5:
            label = "PNEUMONIA"
            conf = prob
        else:
            label = "NORMAL"
            conf = 1 - prob
        st.markdown(f"### Prediction: **{label}**")
        st.write(f"Confidence: {conf:.3f}")
        # Advice/warning
        st.warning("This model is for educational/demo purposes only. Not a medical diagnosis.")
