from src.Green_House_Crop.config.configuration import ConfigurationManager
from src.Green_House_Crop.components.data_validation import DataValiadtion
from src.Green_House_Crop.exception import CustomException
from src.Green_House_Crop.logger import logging
import sys

STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_validation_p(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValiadtion(config=data_validation_config)
            data_validation.validate_all_columns()

        except Exception as e:
            raise CustomException(e, sys)