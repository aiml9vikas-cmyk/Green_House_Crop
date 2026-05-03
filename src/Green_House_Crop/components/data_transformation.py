
import sys
from src.Green_House_Crop.exception import CustomException
from src.Green_House_Crop.logger import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from src.Green_House_Crop.entity.config_entity import DataTransformationConfig

import os

from src.Green_House_Crop.utils.common import save_object

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def get_data_transformer_object(self,df):
        """this function responsible for data transformation"""

        try:
            numerical_columns = df.select_dtypes(exclude='object').columns
            categorical_columns = df.select_dtypes(include='object').columns
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_peplines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise e
     

    def initiate_data_transformation(self):
    
        try:

            raw_data = pd.read_csv(self.config.raw_data_path)
            logging.info(raw_data.shape)
            print(raw_data.shape," raw data")
            raw_data= raw_data.drop_duplicates().reset_index(drop=True)
            raw_data = raw_data.drop(columns=['planting_date','harvest_date'],axis=1,errors='ignore')
            


            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(raw_data,test_size=0.2,random_state=42)

            train_set.to_csv(self.config.train_data_path,index=False,header=True)

            test_set.to_csv(self.config.test_data_path,index=False,header=True)

            logging.info("Train test split completed")



            train_df=pd.read_csv(self.config.train_data_path)
            

            test_df=pd.read_csv(self.config.test_data_path)
            
            logging.info("Splited data into training and test sets")
            logging.info(raw_data.shape)
            logging.info(train_df.shape)
            logging.info(test_df.shape)

            print(raw_data.shape," data after duplicate remove")
            print(train_df.shape," training data")
            print(test_df.shape," testing data")

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

           

            target_column_name="yield_kg_per_m2"
           # numerical_columns = raw_data.select_dtypes(include='number').columns

            

            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

           # input_feature_train_df = input_feature_train_df.drop(columns=['planting_date','harvest_date'], axis=1) 
          #  input_feature_test_df = input_feature_test_df.drop(columns=['planting_date','harvest_date'], axis=1)

            preprocessing_obj=self.get_data_transformer_object(input_feature_train_df)
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.config.pre_processing,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.config.pre_processing,
            )
        except Exception as e:
            raise e


