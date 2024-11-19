import os
import sys
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from hate.logger import logging
import tensorflow as tf
from hate.exception import CustomException
from tensorflow.keras.layers import SpatialDropout1D


class PredictionPipeline:
    def __init__(self, model_directory: str = None, tokenizer_path: str = None):
        self.model_directory = model_directory or os.path.join('model', 'model.h5')
        self.tokenizer_path = tokenizer_path or os.path.join('model', 'tokenizer.pickle')
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the local Keras model."""
        if self.model is None:
            try:
                logging.info(f"Loading the Keras model from {self.model_directory}")
                logging.info(f'Tensorflow version: {tf.__version__}")')
                if not os.path.exists(self.model_directory):
                    raise CustomException(f"Model file not found at {self.model_directory}", sys)

                custom_objects = {'SpatialDropout1D': SpatialDropout1D}

                # Load model with custom objects
                self.model = tf.keras.models.load_model('model/model.h5', custom_objects=custom_objects)
                logging.info("Model loaded successfully")

            except TypeError as te:
                logging.error(f"Error during model deserialization: {te}")
                raise CustomException(f"Model loading error: {str(te)}", sys)
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                raise CustomException(f"Error loading model: {str(e)}", sys)
        return self.model

    def load_tokenizer(self):
        """Load the tokenizer from the specified path."""
        if self.tokenizer is None:
            try:
                logging.info("Loading the tokenizer")
                if not os.path.exists(self.tokenizer_path):
                    raise CustomException(f"Tokenizer not found at {self.tokenizer_path}", sys)

                with open(self.tokenizer_path, 'rb') as handle:
                    self.tokenizer = pickle.load(handle)
                logging.info("Tokenizer loaded successfully")
            except FileNotFoundError as e:
                raise CustomException(f"Tokenizer file not found: {str(e)}", sys)
            except Exception as e:
                raise CustomException(f"Error loading tokenizer: {str(e)}", sys)
        return self.tokenizer

    def preprocess_text(self, text):
        """Clean, tokenize, and pad the input text."""
        try:
            logging.info("Preprocessing the input text")
            tokenizer = self.load_tokenizer()
            tokenized_text = tokenizer.texts_to_sequences([text])
            
            # Ensure padding length matches model input requirements
            padded_text = pad_sequences(tokenized_text, maxlen=300)  # Adjust maxlen if needed
            logging.info(f"Text tokenized and padded. Shape: {padded_text.shape}")
            return padded_text
        except Exception as e:
            raise CustomException(f"Error preprocessing text: {str(e)}", sys)

    def predict(self, text, threshold=0.5):
        """Predict whether the input text is hateful."""
        try:
            logging.info("Starting prediction process")
            model = self.load_model()
            processed_text = self.preprocess_text(text)

            # Perform prediction and ensure it's a scalar value
            prediction = model.predict(processed_text)
            prediction_value = prediction[0] if isinstance(prediction, np.ndarray) else prediction

            logging.info(f"Prediction result: {prediction_value}")

            # Interpret prediction
            if prediction_value > threshold:
                logging.info("Detected hateful content")
                return "Hateful and Abusive"
            else:
                logging.info("Detected non-hateful content")
                return "Not Hateful"
        except Exception as e:
            raise CustomException(f"Error during prediction: {str(e)}", sys)
