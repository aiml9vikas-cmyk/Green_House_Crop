
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
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer
import os
from src.Green_House_Crop.utils.common import save_object

def drop_unnecessary_columns(df):
        # Ensure it returns a DataFrame to keep column names for the next steps
        return df.drop(columns=['planting_date', 'harvest_date'], axis=1, errors='ignore')

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
                    ("imputer",KNNImputer(n_neighbors=5, weights="distance")),
                    ("scalar",StandardScaler())
                    
                ],memory="cache_folder"
            )

            cat_pipeline=Pipeline(

                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    # FIXED: Added handle_unknown='ignore' to prevent crashes
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler",StandardScaler(with_mean=False))
                ],memory="cache_folder"
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # 2. Identify columns (do this AFTER dropping in your logic)
            # Or better: use the config-driven list we discussed earlier
            df_dropped = drop_unnecessary_columns(df)
            numerical_columns = df_dropped.select_dtypes(exclude='object').columns
            categorical_columns = df_dropped.select_dtypes(include='object').columns

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_peplines",cat_pipeline,categorical_columns)
                ]
            )

             # 4. WRAP EVERYTHING in a final pipeline that includes the DROP step
            final_preprocessor = Pipeline(steps=[
                ("drop_cols", FunctionTransformer(drop_unnecessary_columns)),
                ("process", preprocessor)
            ],memory="cache_folder")
        
            return final_preprocessor

        except Exception as e:
            raise CustomException(e,sys)
     

    def initiate_data_transformation(self):
    
        try:

            raw_data = pd.read_csv(self.config.raw_data_path)
            logging.info(raw_data.shape)
           
            raw_data= raw_data.drop_duplicates().reset_index(drop=True)
            
            logging.info("Train test split initiated")
            train_df,test_df=train_test_split(raw_data,test_size=0.2,random_state=42)

            train_df.to_csv(self.config.train_data_path,index=False,header=True)

            test_df.to_csv(self.config.test_data_path,index=False,header=True)

            logging.info("Train test split completed")
            
            logging.info("Splited data into training and test sets")
            logging.info(raw_data.shape)
            logging.info(train_df.shape)
            logging.info(test_df.shape)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

           

            target_column_name=self.config.target_column
         
            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]
        
            preprocessing_obj=self.get_data_transformer_object(input_feature_train_df)
            logging.info(
                f"Applying preprocessing object on training dataframe."
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
            raise CustomException(e,sys)


