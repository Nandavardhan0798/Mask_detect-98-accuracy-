import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# --- Load model ---
@st.cache_resource  # Cache model to avoid reloading on every interaction
def load_mask_model():
    model = load_model("mask_detect.keras")  # Make sure your .keras model is in the same folder
    return model

model = load_mask_model()

# --- App title ---
st.title("Face Mask Detection")
st.write("Upload an image and the model will predict if a person is wearing a mask or not.")

# --- Image upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)


    # Preprocess image
    img = img.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    pred_value = prediction[0][0]

    # Display result
    if pred_value > 0.5:
        st.success("Prediction: **Without Mask** 😷")
    else:
        st.success("Prediction: **With Mask** ✅")

    st.write(f"Prediction confidence: {pred_value:.2f}")
