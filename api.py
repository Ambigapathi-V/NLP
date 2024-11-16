from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import sys
from keras.utils import pad_sequences
import tensorflow as tf
from hate.logger import logging
from hate.exception import CustomException
from hate.pipeline.train_pipeline import TrainPipeline
import mlflow
import os

dagshub_token = os.getenv("DAGSHUB_TOKEN")
if dagshub_token is None:
    raise Exception("Dagshub token is not set in the environment variables.")


train = TrainPipeline()

# Initialize FastAPI app
app = FastAPI()

# Load tokenizer and model
with open("model/tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)
    

logged_model = 'runs:/448f80aaaf804fe19d3aa81c7bbd51ac/model'




# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)



model = tf.keras.models.load_model("artifacts/ModelTrainerArtifacts/model.h5")

logging.info("Model loaded successfully.")
# Define maximum length for padding
MAX_LEN = 300  # Adjust based on your model's configuration

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
        # Preprocess the input text by converting it to sequences
        sequences = tokenizer.texts_to_sequences([text])
        logging.info(f"Text sequence completed: {sequences}")

        # Pad the sequences to ensure consistent input length for the model
        sequences_matrix = pad_sequences(sequences, maxlen=MAX_LEN)
        logging.info(f"Padded sequence completed: {sequences_matrix}")
        
        prediction= loaded_model.predict(sequences_matrix)
        logging.info(f"Prediction completed: {prediction}")


        # Convert predictions to binary labels ('Offensive Language' or 'Hate Speech')
        res = ['Offensive Language' if pred[0] >= 0.5 else 'Hate Speech' for pred in prediction]

        # Return the first prediction result
        return res[0]
    except Exception as e:
        # Log any errors that occur during prediction
        logging.error(f"Error during prediction: {e}")
        # Raise a custom exception with error details
        raise CustomException(e, sys)

# Routes
@app.get("/home")
async def root():
    return {"message": "Welcome to the Hate Speech Detection API!"}

@app.post("/predict")
async def predict(request: TextRequest):
    try:
        # Get text from request
        text = request.text
        # Perform prediction
        prediction = predict_text(text)
        # Return prediction result
        return {"prediction": prediction}  # Return scalar directly
    except CustomException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/train")
async def train_model():
    try:
        train.run_pipeline()
    except CustomException as e:
        raise CustomException(e, sys)