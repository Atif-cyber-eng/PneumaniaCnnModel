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
MODEL_PATH = 'covid_detection.keras'

if not os.path.exists(MODEL_PATH):
    st.write("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

# -------------------------
# 2. Load the model (cached to prevent reloading)
# -------------------------
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_my_model()

# -------------------------
# 3. Label dictionary
# -------------------------
label_dict = {
    0: 'PNEUMONIA',
    1: 'NORMAL'
}

# -------------------------
# 4. Streamlit app layout
# -------------------------
st.title('X-Ray Image Classifier: PNEUMONIA vs. NORMAL')
st.write('Upload an X-ray image to get a prediction on whether it indicates PNEUMONIA or is NORMAL.')

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_data = uploaded_file.read()
    image = Image.open(io.BytesIO(image_data))
    st.image(image, caption='Uploaded X-ray Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess image (resize to 224x224 as model expects)
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    # Make prediction
    predictions = model.predict(img_array)

    # Get predicted label
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_label = label_dict[predicted_class_index]

    # Display probabilities
    if predictions.shape[1] == 2:  # If model outputs [prob_pneumonia, prob_normal]
        st.write(f"Prediction Probabilities: Pneumonia: {predictions[0][0]:.2f}, Normal: {predictions[0][1]:.2f}")
    else:  # If model outputs single probability (sigmoid)
        st.write(f"Prediction Probability: Pneumonia: {predictions[0][0]:.2f}")

    st.success(f"The model predicts: **{predicted_label}**")
