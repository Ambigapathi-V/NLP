from hate.logger import logging
from hate.exception import CustomException
import sys
from hate.configuration.dagshub_sync import download_files

# Create a logger instance
download_files(files_to_download=['data/raw_data.csv', 'data/imbalanced_data.csv'],download_directory='raw')
logging.info("Files downloaded successfully")



