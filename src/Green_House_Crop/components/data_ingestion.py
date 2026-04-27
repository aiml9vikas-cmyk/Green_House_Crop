import os
import urllib.request as request
from src.Green_House_Crop.exception import CustomException
from src.Green_House_Crop.logger import logging
from src.Green_House_Crop.entity.config_entity import (DataIngestionConfig)
import pandas as pd
from sklearn.model_selection import train_test_split


## component-Data Ingestion

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.ingestion_config=config
    
    # Downloading the zip file
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv(r'notebook\greenhouse_crop_yields.csv')
           
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise e
        

