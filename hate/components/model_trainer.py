import os
import sys
import pickle
import pandas as pd
import mlflow
from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from sklearn.model_selection import train_test_split
from hate.configuration.dagshub_sync import dagshub_sync
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import ModelTrainerArtifacts, DataTransformationArtifacts
from hate.ml.model import ModelArchitecture
import dagshub

dagshub.init(repo_owner='Ambigapathi-V', repo_name='NLP', mlflow=True)

class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def spliting_data(self, csv_path):
        try:
            logging.info("Entered the spliting_data function")
            logging.info("Reading the data")
            df = pd.read_csv(csv_path, index_col=False)
            logging.info("Splitting the data into x and y")
            x = df[TWEET]
            y = df[LABEL]

            logging.info("Applying train_test_split on the data")
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
            logging.info(f"Train size: {len(x_train)}, Test size: {len(x_test)}")
            logging.info(f"Train x and y types: {type(x_train)}, {type(y_train)}")
            logging.info("Exited the spliting_data function")
            return x_train, x_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys) from e

    def tokenizing(self, x_train):
        try:
            logging.info("Applying tokenization on the training data")

            # Ensure that all text data in x_train are strings (handle NaN and non-string types)
            x_train = x_train.astype(str)

            # Check for any missing or non-string data
            x_train = x_train.fillna("")

            logging.info(f"Data types in x_train: {x_train.dtypes}")

            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(x_train)
            logging.info("Tokenization complete, converting text to sequences")
            sequences = tokenizer.texts_to_sequences(x_train)
            logging.info(f"Text converted to sequences. Example: {sequences[:5]}")

            logging.info("Padding sequences to ensure consistent input length")
            sequences_matrix = pad_sequences(sequences, maxlen=self.model_trainer_config.MAX_LEN)
            logging.info(f"Sequences padded. Example padded sequence: {sequences_matrix[:5]}")

            return sequences_matrix, tokenizer

        except Exception as e:
            logging.error(f"Error occurred during tokenization: {str(e)}")
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            logging.info("Entered the initiate_model_trainer function ")
            x_train, x_test, y_train, y_test = self.spliting_data(csv_path=self.data_transformation_artifacts.transformed_data_path)
            model_architecture = ModelArchitecture()   
            model = model_architecture.get_model()

            logging.info(f"Xtrain size is : {x_train.shape}")
            logging.info(f"Xtest size is : {x_test.shape}")

            sequences_matrix, tokenizer = self.tokenizing(x_train)
            import dagshub
            dagshub.init(repo_owner='Ambigapathi-V', repo_name='NLP', mlflow=True)
            # Start MLflow logging
            mlflow.start_run()

            # Log model parameters (optional)
            mlflow.log_param("batch_size", self.model_trainer_config.BATCH_SIZE)
            mlflow.log_param("epochs", self.model_trainer_config.EPOCH)
            mlflow.log_param("validation_split", self.model_trainer_config.VALIDATION_SPLIT)
            mlflow.log_param("max_words", self.model_trainer_config.MAX_WORDS)
            mlflow.log_param("max_len", self.model_trainer_config.MAX_LEN)

            logging.info("Entered into model training")
            history = model.fit(sequences_matrix, y_train, 
                                batch_size=self.model_trainer_config.BATCH_SIZE, 
                                epochs=self.model_trainer_config.EPOCH, 
                                validation_split=self.model_trainer_config.VALIDATION_SPLIT)

            # Log metrics
            for epoch in range(self.model_trainer_config.EPOCH):
                mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
                if 'val_loss' in history.history:
                    mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
                if 'val_accuracy' in history.history:
                    mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)

            logging.info("Model training finished")

            # Saving tokenizer
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)

            logging.info("Saving the model")
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)

            # Save model to MLflow
            artifact_path = "artifacts/model_trainer/"
            os.makedirs(artifact_path, exist_ok=True)
            mlflow.keras.log_model(model, "model")
            dagshub_sync.upload_files(local_directory = artifact_path, bucket_name = 'NLP',)
            #dagshub.upload_to_bucket(local_directory = artifact_path, bucket_name = 'NLP')
            

            

            # Saving x_train, y_train, x_test, y_test
            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)
            x_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH)

            # Creating artifacts for the model trainer
            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path=self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path=self.model_trainer_config.Y_TEST_DATA_PATH)
            
            logging.info("Returning the ModelTrainerArtifacts")

            # End MLflow run
            mlflow.end_run()

            return model_trainer_artifacts

        except Exception as e:
            logging.error(f"Error occurred during model training: {str(e)}")
            raise CustomException(e, sys) from e
