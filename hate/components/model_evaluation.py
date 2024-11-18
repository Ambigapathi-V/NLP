import os
import sys
import keras
import pickle
import numpy as np
import pandas as pd
from hate.logger import logging
from hate.exception import CustomException
from keras.utils import pad_sequences
from hate.constants import *
from hate.configuration.dagshub_sync import dagshub_sync
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import mlflow
import dagshub
dagshub.init(repo_owner='Ambigapathi-V', repo_name='NLP', mlflow=True)
from hate.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts


class ModelEvaluation:
    def __init__(self, 
                 model_trainer_artifacts: ModelTrainerArtifacts,
                 data_transformation_artifacts: DataTransformationArtifacts):
        """
        :param model_evaluation_config: Configuration for model evaluation
        :param model_trainer_artifacts: Output reference of model trainer artifact stage
        :param data_transformation_artifacts: Data transformation artifact stage
        """
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        self.dagshub_sync = dagshub_sync()

    def evaluate(self):
        try:
            logging.info("Entering into the evaluate function of Model Evaluation class")
            print(self.model_trainer_artifacts.x_test_path)

            # Loading test data
            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path, index_col=0)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path, index_col=0)

            # Loading tokenizer
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)

            # Loading the trained model
            load_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)

            x_test = x_test['tweet'].astype(str)
            x_test = x_test.squeeze()
            y_test = y_test.squeeze()

            # Tokenizing and padding test sequences
            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN)

            # Evaluating model performance
            accuracy, loss = load_model.evaluate(test_sequences_matrix, y_test)
            logging.info(f"Test accuracy: {accuracy}")

            # Making predictions for additional metrics
            predictions = load_model.predict(test_sequences_matrix)
            res = [1 if pred[0] >= 0.5 else 0 for pred in predictions]

            # Calculate performance metrics
            precision = precision_score(y_test, res)
            recall = recall_score(y_test, res)
            f1 = f1_score(y_test, res)
            roc_auc = roc_auc_score(y_test, res)
            cm = confusion_matrix(y_test, res)
            cr = classification_report(y_test, res)

            # Log metrics to MLFlow
            with mlflow.start_run() as run:
                mlflow.log_metric("test_accuracy", accuracy)
                mlflow.log_metric("test_loss", loss)
                mlflow.log_metric("test_precision", precision)
                mlflow.log_metric("test_recall", recall)
                mlflow.log_metric("test_f1_score", f1)
                mlflow.log_metric("test_roc_auc", roc_auc)
                mlflow.log_metric("test_confusion_matrix", (cm))  # Convert cm to string to log
                mlflow.log_metric("test_classification_report", cr)

            # Returning accuracy
            return accuracy

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Initiating Model Evaluation")
        try:
            logging.info("Loading currently trained model")
            trained_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)

            # Loading tokenizer
            with open('tokenizer.pickle', 'rb') as handle:
                load_tokenizer = pickle.load(handle)

            # Get accuracy of the trained model
            trained_model_accuracy = self.evaluate()

            # For best model evaluation (you can define your criteria for best model)
            best_model_accuracy = self.evaluate()
            logging.info(f"Best model accuracy: {best_model_accuracy}")
            logging.info(f"Trained model accuracy: {trained_model_accuracy}")
            logging.info("Model evaluation completed")

        except Exception as e:
            raise CustomException(e, sys) from e
