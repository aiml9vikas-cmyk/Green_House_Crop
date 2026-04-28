from src.Green_House_Crop.config.configuration import ConfigurationManager
from src.Green_House_Crop.components.data_transformation import DataTransformation
from src.Green_House_Crop.logger import logging

from pathlib import Path


STAGE_NAME="Data Trnasformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation_p(self):

        try:
            with open(Path("artifacts/data_validation/status.txt"),'r') as f:
                status=f.read().split(" ")[-1]
            if status=="True":
                config=ConfigurationManager()
                data_transformation_config=config.get_data_transformation_config()
                data_transformation=DataTransformation(config=data_transformation_config)
                data_transformation.initiate_data_transformation()
            else:
                raise Exception("Your data scheme is not valid")
            
        except Exception as e:
            print(e)