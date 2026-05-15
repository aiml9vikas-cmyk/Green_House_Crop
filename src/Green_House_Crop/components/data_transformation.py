
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA  # Added PCA for dimensionality reduction

class GrowthDurationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, planting_col='planting_date', harvest_col='harvest_date'):
        self.planting_col = planting_col
        self.harvest_col = harvest_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        # Convert to datetime and calculate duration in days
        if self.planting_col in X_df.columns and self.harvest_col in X_df.columns:
            p_date = pd.to_datetime(X_df[self.planting_col])
            h_date = pd.to_datetime(X_df[self.harvest_col])
            X_df['growth_duration_days'] = (h_date - p_date).dt.days
        
        # Now it is safe to drop the original date strings
        X_df = X_df.drop(columns=[self.planting_col, self.harvest_col], errors='ignore')
        return X_df

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bound = {}
        self.upper_bound = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for column in X_df.columns:
            Q1 = X_df[column].quantile(0.25)
            Q3 = X_df[column].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bound[column] = Q1 - self.factor * IQR
            self.upper_bound[column] = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for column in X_df.columns:
            X_df[column] = np.clip(X_df[column], self.lower_bound[column], self.upper_bound[column])
        return X_df.values

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
                    ("outlier_handler", OutlierCapper(factor=1.5)),
                    ("scalar",StandardScaler())
                    
                ],memory="cache_folder"
            )

            cat_pipeline=Pipeline(

                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    # FIXED: Added handle_unknown='ignore' to prevent crashes
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
                    ("scaler",StandardScaler(with_mean=False))
                ],memory="cache_folder"
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

           
            df_processed = GrowthDurationTransformer().transform(df)
            numerical_columns = df_processed.select_dtypes(exclude='object').columns
            categorical_columns = df_processed.select_dtypes(include='object').columns

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_peplines",cat_pipeline,categorical_columns)
                ]
            )

             # 4. WRAP EVERYTHING in a final pipeline that includes the DROP step
            final_preprocessor = Pipeline(steps=[
                ("date_engineering", GrowthDurationTransformer()),
                ("process", preprocessor),
                # n_components=0.95 selects minimum components needed to keep 95% variance
                #("pca", PCA(n_components=0.99, random_state=42))      
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


