import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.layers import Rescaling

# Load model
@st.cache_resource
def load_model():
    custom_objects = {
        "MobileNetV3Small": MobileNetV3Small,
        "Rescaling": Rescaling
    }
    return tf.keras.models.load_model("mobilenetv3_crop_disease_model.keras", custom_objects=custom_objects)

model = load_model()

# Load class labels from file
with open("class_labels.txt", "r") as f:
    class_labels = [line.strip() for line in f]

# Streamlit UI
st.title("🌿 Crop Disease and Pest Detection")
st.write("Upload an image of a crop leaf to detect possible diseases or pest damage.")

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_container_width=True)


    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)  # Same preprocessing as training
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx] * 100

    st.success(f"Prediction: **{class_labels[class_idx]}** with **{confidence:.2f}%** confidence.")
#project by light of GOD