import os
import sys
from src.Green_House_Crop.exception import CustomException
from src.Green_House_Crop.logger import logging
from src.Green_House_Crop.entity.config_entity import (DataIngestionConfig)
import pandas as pd


class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.ingestion_config=config
    
    # Downloading the zip file
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(r'notebook\greenhouse_crop_yields.csv')
           
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            return(
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        

