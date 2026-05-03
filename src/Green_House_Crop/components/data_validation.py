from src.Green_House_Crop.exception import CustomException
from src.Green_House_Crop.logger import logging
import pandas as pd
from src.Green_House_Crop.entity.config_entity import DataValidationConfig
import sys

class DataValiadtion:

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = True  # Start assuming it's valid
            data = pd.read_csv(self.config.raw_data_path)
            all_cols = list(data.columns)
            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    logging.error(f"Column {col} not found in schema!")
                    break  # Stop immediately if one column fails
                
            # Write the final status once after the loop
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise CustomException(e, sys)


"""
    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.raw_data_path)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise CustomException(e,sys)
"""
    

