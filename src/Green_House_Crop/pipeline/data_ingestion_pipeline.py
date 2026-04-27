import logging
import logging
from src.Green_House_Crop.config.configuration import ConfigurationManager
from src.Green_House_Crop.components.data_ingestion import DataIngestion
from src.Green_House_Crop.exception import CustomException
from src.Green_House_Crop.logger import logging
import sys


STAGE_NAME="Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion_p(self):
        config=ConfigurationManager()
        data_ingestion_config=config.get_data_ingestion_config()
        data_ingestion=DataIngestion(config=data_ingestion_config)
        data_ingestion.initiate_data_ingestion()
        
