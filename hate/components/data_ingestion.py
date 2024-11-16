import os
import sys
from zipfile import ZipFile
from hate.logger import logging
from hate.exception import CustomException
from hate.configuration.dagshub_sync import dagshub_sync
from hate.entity.config_entity import DataIngestionConfig
from hate.entity.artifact_entity import DataIngestionArtifacts

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def get_data_from_bucket(self)-> None:
        try:
            logging.info('Downloading data from S3 bucket')
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)
            dagshub_sync.download_file(download_directory=self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)
            logging.info('Data downloaded successfully')
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def unzip_and_clean(self):
        try:
            logging.info('Unzipping and cleaning data...')
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)
                logging.info('Data unzipped and cleaned successfully')
                return self.data_ingestion_config.DATA_ARTIFACTS_DIR,self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR
        except Exception as e:
            raise CustomException(e, sys) from e
        
    
        
    def initiate_data_ingestion(self) ->DataIngestionArtifacts:
        logging.info('Data Ingestion initiated')
        try:
            self.get_data_from_bucket()
            imbalance_data_file_path, raw_data_file_path = self.unzip_and_clean()
            data_ingestion_artifacts = DataIngestionArtifacts(
                imblanced_data_path=imbalance_data_file_path, 
                raw_data_file_path=raw_data_file_path
            )
            
            logging.info('Data Ingestion completed successfully')
            logging.info(f'Artifacts: {data_ingestion_artifacts}')
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e

