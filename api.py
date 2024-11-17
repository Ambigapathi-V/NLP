from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import tensorflow as tf
from keras.utils import pad_sequences
import mlflow
import sys
from hate.logger import logging
from hate.exception import CustomException
from hate.pipeline.train_pipeline import TrainPipeline

# Directly disable CUDA (GPU usage)
tf.config.set_visible_devices([], 'GPU')  # Ensure TensorFlow does not try to use the GPU
logging.info("🚫 CUDA (GPU) disabled. Running TensorFlow on CPU.")

# Hardcoded DagsHub token for authentication
DAGSHUB_TOKEN = "your-hardcoded-dagshub-token"
if not DAGSHUB_TOKEN:
    raise Exception("❗ DagsHub token is missing. Set your token in the script.")

# TrainPipeline instance
train = TrainPipeline()

# Initialize FastAPI app
app = FastAPI()

# Load tokenizer
try:
    with open("model/tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    logging.info("✅ Tokenizer loaded successfully.")
except FileNotFoundError:
    logging.error("❌ Tokenizer file not found.")
    raise Exception("Tokenizer file is missing.")

# Load TensorFlow model
try:
    model = tf.keras.models.load_model("artifacts/ModelTrainerArtifacts/model.h5")
    logging.info("✅ TensorFlow model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load TensorFlow model: {e}")
    model = None

# Define maximum length for padding
MAX_LEN = 300  # Adjust this as per your model configuration

# Input data model
class TextRequest(BaseModel):
    text: str

# Prediction function
def predict_text(text):
    """
    Predicts whether the given text contains 'Offensive Language' or 'Hate Speech'.

    :param text: The input text to be classified.
    :return: A string indicating 'Offensive Language' or 'Hate Speech'.
    """
    try:
        if not model:
            raise Exception("No TensorFlow model is loaded for predictions.")
        # Preprocess the input text
        sequences = tokenizer.texts_to_sequences([text])
        sequences_matrix = pad_sequences(sequences, maxlen=MAX_LEN)

        # Get prediction
        prediction = model.predict(sequences_matrix)
        logging.info(f"Prediction result: {prediction}")

        # Convert predictions to labels
        return "Offensive Language" if prediction[0] >= 0.5 else "Hate Speech"

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise CustomException(e, sys)

# Routes
@app.get("/home")
async def root():
    return {"message": "Welcome to the Hate Speech Detection API!"}

@app.post("/predict")
async def predict(request: TextRequest):
    """
    API endpoint for making predictions.

    :param request: Input text wrapped in the TextRequest model.
    :return: Predicted label for the text.
    """
    try:
        text = request.text
        prediction = predict_text(text)
        return {"prediction": prediction}
    except CustomException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/train")
async def train_model():
    """
    API endpoint to trigger training pipeline.

    :return: Status of the training process.
    """
    try:
        train.run_pipeline()
        return {"message": "Training pipeline executed successfully!"}
    except CustomException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error during training: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
