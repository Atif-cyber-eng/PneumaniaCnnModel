# app.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os
import io
import traceback

st.set_page_config(page_title="Pneumonia X-ray Detector", layout="centered")

st.title("Pneumonia Detector — Chest X-ray (Demo)")
st.markdown(
    """
This demo loads a Keras `.h5` model and predicts **PNEUMONIA** vs **NORMAL** on an uploaded chest X-ray.
**Important:** if the app cannot import TensorFlow (heavy dependency) you'll see instructions below — follow them to fix deployment.
"""
)

MODEL_PATH = "https://drive.google.com/file/d/12XyA6c8ykWGpO5U1eUUCri963BKfsIFg/view?usp=sharing"  # change if using a different path
GDRIVE_FILE_ID = "12XyA6c8ykWGpO5U1eUUCri963BKfsIFg"  # optional: if you host model on Google Drive, put file id here

# --- Utility functions ---

def show_dependency_help():
    st.error(
        """
        The app couldn't import TensorFlow or otherwise failed to prepare the ML runtime.
        Common fixes:
        1. In `requirements.txt` use `tensorflow-cpu==2.12.0` (or a supported CPU build) instead of `tensorflow`.
        2. If the model is large (>50MB) don't store it in the repo — host it (Google Drive, S3) and use a downloader in the app.
        3. If you can't install TensorFlow on Streamlit Cloud, host the model in a small inference server (Heroku/AWS Lambda/GCF) and call it via HTTP.
        After making changes push an update to GitHub and redeploy.
        """
    )

def preprocess_image(img: Image.Image, target_size=(224,224)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = ImageOps.fit(img, target_size, Image.ANTIALIAS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@st.cache_resource(show_spinner=False)
def download_model_from_drive(file_id: str, target_path: str):
    """
    Try to download a file from a public Google Drive file-id using gdown.
    Returns True if file exists after the call.
    """
    if os.path.exists(target_path):
        return True, "already_exists"
    try:
        import gdown
    except Exception as e:
        return False, f"gdown import failed: {e}"
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, target_path, quiet=False)
        return os.path.exists(target_path), "download_attempted"
    except Exception as e:
        return False, f"download failed: {e}"

@st.cache_resource(show_spinner=False)
def load_model_lazy(model_path: str):
    """
    Lazy-load the Keras model. Returns (model, None) on success or (None, error_msg) on failure.
    """
    try:
        # Import here to avoid failing app build if tensorflow is not installable
        import tensorflow as tf
        from tensorflow.keras.models import load_model
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"TensorFlow import failed: {e}\n\nTraceback:\n{tb}"

    if not os.path.exists(model_path):
        return None, f"Model file not found at {model_path}"

    try:
        model = load_model(model_path)
        return model, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"Failed loading model from {model_path}: {e}\n\nTraceback:\n{tb}"

# --- Try auto-download if user provided a Drive file id ---
if GDRIVE_FILE_ID and not os.path.exists(MODEL_PATH):
    ok, info = download_model_from_drive(GDRIVE_FILE_ID, MODEL_PATH)
    if not ok:
        st.info(f"Model download not completed: {info}")
    else:
        st.success("Model downloaded from Google Drive.")

# Show model status
if os.path.exists(MODEL_PATH):
    st.success(f"Model found at `{MODEL_PATH}`")
else:
    st.warning(f"Model not found at `{MODEL_PATH}`. Upload a model to the repo or set `GDRIVE_FILE_ID` and redeploy.")
    st.markdown("If you don't have the model in the repo, you can upload it to Google Drive and set `GDRIVE_FILE_ID` in the code.")

# Provide a file uploader to upload a model at runtime (helpful during testing)
st.markdown("### (Optional) Upload a `pneumonia_model.h5` here (for testing without redeploy):")
uploaded_model_file = st.file_uploader("Upload model (.h5)", type=["h5"], accept_multiple_files=False, key="model")
if uploaded_model_file is not None:
    try:
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded_model_file.read())
        st.success(f"Saved uploaded model to `{MODEL_PATH}`. You can now load it below.")
    except Exception as e:
        st.error(f"Failed to save uploaded model: {e}")

# Image upload / prediction block
st.markdown("---")
st.markdown("### Upload chest X-ray image for prediction")
uploaded_image = st.file_uploader("Upload X-ray (jpg/png)", type=["jpg","jpeg","png"], accept_multiple_files=False, key="img")

if uploaded_image is not None:
    try:
        img = Image.open(io.BytesIO(uploaded_image.read()))
        st.image(img, caption="Uploaded image", use_column_width=True)
    except Exception as e:
        st.error(f"Cannot open image: {e}")
        img = None
else:
    img = None

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Predict"):
        if img is None:
            st.warning("Upload an image first.")
        else:
            # Lazy-load model
            with st.spinner("Loading model and running prediction..."):
                model, err = load_model_lazy(MODEL_PATH)
                if err is not None:
                    st.error("Model load / dependency error:")
                    st.code(err)
                    show_dependency_help()
                else:
                    try:
                        X = preprocess_image(img, target_size=(224,224))
                        pred_prob = model.predict(X)[0].ravel()
                        # handle both shapes (N,1) or (N,)
                        if hasattr(pred_prob, "__len__"):
                            prob = float(pred_prob[0])
                        else:
                            prob = float(pred_prob)
                        threshold = 0.5
                        if prob >= threshold:
                            label = "PNEUMONIA"
                            conf = prob
                        else:
                            label = "NORMAL"
                            conf = 1 - prob
                        st.success(f"Prediction: **{label}**")
                        st.write(f"Confidence: {conf:.3f}")
                        st.warning("This is a demo model — not a medical diagnosis.")
                    except Exception as e:
                        tb = traceback.format_exc()
                        st.error(f"Prediction failed: {e}")
                        st.code(tb)

with col2:
    st.markdown("### Troubleshooting / Logs")
    st.markdown("- If deployment failed with `installer returned a non-zero exit code`, it's likely a package in `requirements.txt` failed to install.")
    st.markdown("- Common fix: use `tensorflow-cpu==2.12.0` in `requirements.txt` or host the model externally and remove TensorFlow from requirements (call an inference API instead).")
    st.markdown("- You can also upload a model here to test locally in the running app (won't persist across restarts).")
    st.markdown("### Quick commands to try locally")
    st.code(
        "pip install -r requirements.txt\nstreamlit run app.py",
        language="bash"
    )
