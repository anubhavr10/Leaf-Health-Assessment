# streamlit_app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle

# Load the model from the pickle file
with open('pickle.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Create a Streamlit app
st.title("Image Classification Web App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the selected image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image for the model
    image = tf.image.resize(np.array(image), (128, 128))
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet.preprocess_input(image)

    # Make predictions using the loaded model
    prediction = loaded_model.predict(image)

    # Display the prediction
    st.subheader("Model Prediction:")
    if prediction[0][0] > 0.5:
        st.write("The model predicts: Diseased")
    else:
        st.write("The model predicts: Healthy")
