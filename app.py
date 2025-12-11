
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# 1. Load the pre-trained covid_detection.keras model
# Ensure the model path is correct or accessible in the Streamlit app environment
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('https://drive.google.com/file/d/15ji-5Y6fwZmEBqw0MlP1Z6L8RAmWKAVh/view?usp=sharing')
    return model

model = load_my_model()

# Define the label dictionary consistent with training
label_dict = {
    0: 'PNEUMONIA',
    1: 'NORMAL'
}

# 2. Set up the Streamlit application title and a brief description
st.title('X-Ray Image Classifier: PNEUMONIA vs. NORMAL')
st.write('Upload an X-ray image to get a prediction on whether it indicates PNEUMONIA or is NORMAL.')

# 3. Add a file uploader widget
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

# 4. Check if an image has been uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, caption='Uploaded X-ray Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the uploaded image
    # Resize to (224, 224) as used in training
    img_resized = image.resize((224, 224))
    # Convert to NumPy array
    img_array = np.array(img_resized)
    # Expand dimensions to (1, 224, 224, 3) to match model input shape
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    predictions = model.predict(img_array)

    # Interpret the prediction
    # The model was trained with to_categorical, so output will be probability for each class
    # assuming binary classification [prob_pneumonia, prob_normal]
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_label = label_dict[predicted_class_index]
    
    # Display probabilities (optional, but good for understanding confidence)
    st.write(f"Prediction Probabilities: Pneumonia: {predictions[0][0]:.2f}, Normal: {predictions[0][1]:.2f}")

    st.success(f"The model predicts: **{predicted_label}**")
