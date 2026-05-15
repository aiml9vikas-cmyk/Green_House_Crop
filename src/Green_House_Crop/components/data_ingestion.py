import os
import sys
from src.Green_House_Crop.exception import CustomException
from src.Green_House_Crop.logger import logging
from src.Green_House_Crop.entity.config_entity import (DataIngestionConfig)
import pandas as pd
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv  # Added to load environment variables


class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.ingestion_config=config

        load_dotenv()
    
    
    # Downloading the zip file
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Fetch variables securely from the environment
            MONGO_URI = os.getenv("MONGO_URI")
            DATABASE_NAME = os.getenv("DATABASE_NAME")
            COLLECTION_NAME = os.getenv("COLLECTION_NAME")

            # Validation check to ensure variables loaded correctly
            if not all([MONGO_URI, DATABASE_NAME, COLLECTION_NAME]):
                raise Exception("Missing required MongoDB environment variables in .env file.")

            logging.info("Connecting to MongoDB Atlas cluster securely via environment variables...")
            
            # Establish connection
            client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
            db = client[DATABASE_NAME]
            collection = db[COLLECTION_NAME]

            logging.info(f"Fetching data from collection: {COLLECTION_NAME}")
            cursor = collection.find()
            data_list = list(cursor)
            
            if len(data_list) == 0:
                raise Exception(f"Database collection '{COLLECTION_NAME}' contains 0 records. Ingestion halted.")

            logging.info(f"Successfully retrieved {len(data_list)} records from MongoDB Atlas.")

            df = pd.DataFrame(data_list)
            
            # Clean metadata
            if '_id' in df.columns:
                df = df.drop(columns=['_id'])
           
            logging.info('Transformed the MongoDB JSON records into a pandas DataFrame.')

            # Create artifacts directory and save data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info(f"Successfully written raw data asset locally at: {self.ingestion_config.raw_data_path}")
            
            client.close()
            return self.ingestion_config.raw_data_path
            
        except Exception as e:
            raise CustomException(e, sys)
        

