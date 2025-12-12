# app.py
import os
import streamlit as st
from PIL import Image
import numpy as np
import io
import tempfile
import sys

# Try to import tensorflow/keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception as e:
    st.write("Error importing TensorFlow. Make sure tensorflow is installed.")
    raise

# Optional: gdown for downloading from Google Drive
try:
    import gdown
    _HAS_GDOWN = True
except Exception:
    _HAS_GDOWN = False

st.set_page_config(page_title="Pneumonia Detection", layout="centered")

st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image (jpg/png). The app will preprocess and run the model to predict Pneumonia vs Normal.")

MODEL_FILENAME = "pneumonia_model.h5"
# If you want the app to download from your Drive, put your file id here:
GDRIVE_FILE_ID = "12XyA6c8ykWGpO5U1eUUCri963BKfsIFg"  # from your share link

@st.cache_resource
def load_the_model(model_path=MODEL_FILENAME):
    if not os.path.exists(model_path):
        # Try to download automatically if gdown available and file id provided
        if _HAS_GDOWN and GDRIVE_FILE_ID:
            st.info("Model not found locally — attempting to download from Google Drive...")
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            try:
                gdown.download(url, model_path, quiet=False)
            except Exception as e:
                st.error("Automatic download failed. Please place the model file 'pneumonia_model.h5' in the app folder.")
                raise
        else:
            raise FileNotFoundError(f"Model file '{model_path}' not found and automatic download unavailable.")
    # Load model
    model = load_model(model_path)
    return model

def preprocess_image(pil_img, target_size):
    # Ensure RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    img = pil_img.resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    # If model expects grayscale / single channel, adjust outside (we infer from model shape)
    return arr

def predict_image(model, img_array):
    # model.input_shape typically (None, H, W, C) or similar
    # make batch
    batch = np.expand_dims(img_array, axis=0)
    preds = model.predict(batch)
    return preds

def interpret_prediction(preds):
    # Preds shape can be (1,1) for sigmoid or (1,2) for softmax, or more.
    if preds.ndim == 2 and preds.shape[1] == 1:
        prob = float(preds[0,0])
        # sigmoid: higher -> class 1
        label = "Pneumonia" if prob >= 0.5 else "Normal"
        confidence = prob if prob >= 0.5 else 1 - prob
        # Provide returned structure
        return {"label": label, "confidence": confidence, "probability": prob, "raw": preds.tolist()}
    elif preds.ndim == 2:
        probs = preds[0].tolist()
        idx = int(np.argmax(probs))
        # Default label order: [Normal, Pneumonia]. If your training used different ordering,
        # change LABELS variable below.
        LABELS = ["Normal", "Pneumonia"]
        if idx < len(LABELS):
            label = LABELS[idx]
        else:
            label = f"Class {idx}"
        confidence = probs[idx]
        return {"label": label, "confidence": float(confidence), "probabilities": probs, "raw": preds.tolist()}
    else:
        # Unexpected shape: return raw
        return {"label": "Unknown", "raw": preds.tolist()}

# Main UI
uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["png", "jpg", "jpeg"])
col1, col2 = st.columns([1,1])

with col1:
    st.write("Model status:")
    try:
        model = load_the_model()
        st.success(f"Loaded model: {MODEL_FILENAME}")
        # Determine expected input size from model
        try:
            input_shape = model.input_shape  # e.g. (None, 224, 224, 3)
            # pick height, width
            if len(input_shape) >= 3:
                # often (None, H, W, C)
                _, h, w, *rest = (None, ) + tuple(input_shape[1:]) if len(input_shape)==4 else (None,) + tuple(input_shape[1:])
                # safer parse:
                if len(input_shape) == 4:
                    _, H, W, C = input_shape
                    target_size = (int(W), int(H))
                elif len(input_shape) == 3:
                    # (None, H, W) or (H, W, C) tricky — fallback to 224
                    target_size = (224, 224)
                else:
                    target_size = (224, 224)
            else:
                target_size = (224, 224)
        except Exception:
            target_size = (224, 224)
        st.write(f"Model input size (inferred): {target_size[0]}x{target_size[1]}")
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Place the model file 'pneumonia_model.h5' in the same folder as this app, or enable gdown and internet access for automatic download.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

with col2:
    st.write("How it works")
    st.write("""
    1. Upload an X-ray image.  
    2. The app resizes the image to the model's expected size and normalizes pixels.  
    3. The model predicts: either sigmoid (single probability) or softmax (class probabilities).  
    """)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error("Could not open image. Try a different file.")
        raise

    st.image(image, caption="Uploaded image", use_column_width=True)
    st.write("Running prediction...")

    # Preprocess using inferred size
    img_arr = preprocess_image(image, target_size)
    # If model expects single channel (grayscale), convert:
    # check model.input_shape channel dim
    try:
        shape = model.input_shape
        # if channel dim is 1, convert to grayscale array with a single channel
        if len(shape) == 4 and int(shape[-1]) == 1:
            # convert to grayscale
            img_gray = image.convert("L").resize(target_size)
            img_arr = np.asarray(img_gray).astype("float32") / 255.0
            img_arr = np.expand_dims(img_arr, axis=-1)
        # else keep RGB
    except Exception:
        pass

    preds = predict_image(model, img_arr)
    result = interpret_prediction(preds)

    st.markdown("### Result")
    if "label" in result:
        st.write(f"**Prediction:** {result['label']}")
    if "confidence" in result:
        st.write(f"**Confidence:** {result['confidence']:.3f}")
    if "probability" in result:
        st.write(f"**Raw probability (sigmoid):** {result['probability']:.4f}")
    if "probabilities" in result:
        # show per-class probs (assumes two classes)
        probs = result["probabilities"]
        st.write("Class probabilities (raw):")
        for i, p in enumerate(probs):
            st.write(f"Class {i}: {p:.4f}")
        st.info("Default class names used: [0]=Normal, [1]=Pneumonia. Change LABELS in the app if your model uses a different order.")
    # Optionally show raw model output for debug
    with st.expander("Show raw model output"):
        st.write(result.get("raw", preds.tolist()))

    st.success("Done")

st.markdown("---")
st.markdown("**Notes & troubleshooting**")
st.markdown("""
- If the model predictions look wrong, the class ordering used during training may differ (change the `LABELS` variable in the script).  
- If the model uses a different image normalization (e.g., mean/std or inputs in range [-1,1]), adjust `preprocess_image`.  
- If loading fails with `BadKeyError` or similar, the model file might be corrupted or saved by a different TF version.
""")
