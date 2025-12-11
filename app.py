import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import gdown
import os

# -------------------------
# 1. Download the model from Google Drive if not already present
# -------------------------
MODEL_URL = 'https://drive.google.com/uc?id=15ji-5Y6fwZmEBqw0MlP1Z6L8RAmWKAVh'
MODEL_PATH = 'pneumonia_model.h5'

if not os.path.exists(MODEL_PATH):
    st.write("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

# -------------------------
# 2. Load the model
# -------------------------
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_my_model()

# -------------------------
# 3. Get model input details
# -------------------------
input_shape = model.input_shape  # e.g., (None, 150, 150, 3)
_, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = input_shape

st.write(f"Model expects input shape: {IMG_HEIGHT}x{IMG_WIDTH} with {IMG_CHANNELS} channels")

# Label dictionary (adjust according to your model training)
label_dict = {0: 'PNEUMONIA', 1: 'NORMAL'}

# -------------------------
# 4. Streamlit layout
# -------------------------
st.title('X-Ray Image Classifier: PNEUMONIA vs. NORMAL')
st.write('Upload an X-ray image to get a prediction on whether it indicates PNEUMONIA or is NORMAL.')

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image_data = uploaded_file.read()
    image = Image.open(io.BytesIO(image_data))
    st.image(image, caption='Uploaded X-ray Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # -------------------------
    # 5. Preprocess image
    # -------------------------
    # Convert to RGB if model expects 3 channels
    if IMG_CHANNELS == 3:
        image = image.convert('RGB')
    elif IMG_CHANNELS == 1:
        image = image.convert('L')  # grayscale

    # Resize image
    img_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))

    # Convert to numpy array
    img_array = np.array(img_resized)

    # Add channel dimension if missing
    if IMG_CHANNELS == 1 and len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)

    # Expand batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize
    img_array = img_array / 255.0

    # -------------------------
    # 6. Make prediction
    # -------------------------
    predictions = model.predict(img_array)

    # -------------------------
    # 7. Decode prediction
    # -------------------------
    if predictions.shape[1] == 1:  # Sigmoid output
        pred_prob = predictions[0][0]
        if pred_prob > 0.5:
            predicted_label = 'PNEUMONIA'
        else:
            predicted_label = 'NORMAL'
        st.write(f"Prediction Probability: Pneumonia: {pred_prob:.2f}")
    else:  # Softmax output
        pred_index = np.argmax(predictions, axis=1)[0]
        predicted_label = label_dict[pred_index]
        st.write(f"Prediction Probabilities: Pneumonia: {predictions[0][0]:.2f}, Normal: {predictions[0][1]:.2f}")

    st.success(f"The model predicts: **{predicted_label}**")
