import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
from hate.logger import logging
from hate.exception import CustomException

# Title of the application
st.title("Hate Speech Detection")

# Disable GPU (optional, for CPU-only inference)
tf.config.set_visible_devices([], 'GPU')
logging.info("🚫 CUDA (GPU) disabled. Running TensorFlow on CPU.")

# Load the pre-trained tokenizer
try:
    with open("model/tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    logging.info("✅ Tokenizer loaded successfully.")
except FileNotFoundError:
    st.error("❌ Tokenizer file not found. Please upload or provide the tokenizer file.")
    raise CustomException("Tokenizer file is missing.")

# Load the pre-trained TensorFlow model
try:
    model = tf.keras.models.load_model("artifacts/ModelTrainerArtifacts/model.h5")
    logging.info("✅ TensorFlow model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load TensorFlow model: {e}")
    raise CustomException(e)

# Define the maximum sequence length for padding
MAX_LEN = 300

# Function for prediction
def predict_text(text):
    """
    Predicts whether the given text contains 'Offensive Language' or 'Hate Speech'.

    :param text: The input text to be classified.
    :return: A string indicating 'Offensive Language' or 'Hate Speech'.
    """
    try:
        # Preprocess the input text
        sequences = tokenizer.texts_to_sequences([text])
        sequences_matrix = pad_sequences(sequences, maxlen=MAX_LEN)

        # Get model prediction
        prediction = model.predict(sequences_matrix)
        logging.debug(f"Prediction result: {prediction}")

        # Convert prediction to label (assuming binary classification)
        return "Offensive Language" if prediction[0] >= 0.5 else "Hate Speech"
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise CustomException(e)

# User input for text
text = st.text_area("Enter text to classify", height=150)

# Predict button
if st.button("Predict"):
    if not text.strip():
        st.warning("⚠️ Please enter some text for prediction.")
    else:
        try:
            with st.spinner("Analyzing..."):
                prediction = predict_text(text)
            st.success(f"The prediction is: {prediction}")
        except CustomException as e:
            st.error(f"An error occurred: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# Batch predictions from a file (optional)
st.markdown("---")
st.header("Batch Predictions (Optional)")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("The uploaded file must contain a 'text' column.")
        else:
            texts = df["text"].tolist()

            # Batch prediction (optimizing for multiple texts at once)
            sequences = tokenizer.texts_to_sequences(texts)
            sequences_matrix = pad_sequences(sequences, maxlen=MAX_LEN)
            predictions = model.predict(sequences_matrix)

            # Assign predictions based on the threshold
            df["Prediction"] = ["Offensive Language" if pred >= 0.5 else "Hate Speech" for pred in predictions]

            # Show results
            st.write(df)
            st.download_button(
                label="Download Predictions",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")

# Footer information
st.markdown("---")
st.markdown("Hate Speech Detection App | Powered by TensorFlow and Streamlit")
