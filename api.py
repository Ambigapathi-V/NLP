from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import tensorflow as tf
from keras.utils import pad_sequences
import mlflow
import os
import sys
from dagshub.auth import basic_auth
from hate.logger import logging
from hate.exception import CustomException
from hate.pipeline.train_pipeline import TrainPipeline

# Validate Dagshub token
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:  
    raise Exception("❗ DAGSHUB_TOKEN is not set in the environment variables.")
basic_auth(username="Ambigapathi-V", password=dagshub_token)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

# Load MLflow model (if using it)
logged_model = 'runs:/448f80aaaf804fe19d3aa81c7bbd51ac/model'
try:
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    logging.info("✅ MLflow model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load MLflow model: {e}")
    loaded_model = None

# Load TensorFlow model (if using it)
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
