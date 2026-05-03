import os
import sys
from src.Green_House_Crop.entity.config_entity import ModelTrainerConfig
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from src.Green_House_Crop.constants import *
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import yaml
from src.Green_House_Crop.exception import CustomException
from src.Green_House_Crop.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.Green_House_Crop.utils.common import save_object

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def evaluate_models(self,X_train, y_train,X_test,y_test,models,param):
        try:
            report = {}
                      
            for model_name, model in models.items():
                # Get params from yaml, default to {} if missing or None
                para = param.get(model_name, {})
                if para is None:
                    para = {}

                logging.info(f"Started training: {model_name}")
                # 1. Find Best Parameters using GridSearch
                gs = GridSearchCV(model,para,cv=3,n_jobs=-1,scoring='r2')
                gs.fit(X_train,y_train)

                # 2. Update the model with the best parameters found
                model.set_params(**gs.best_params_)
                 # 3. Final Fit with Early Stopping (for compatible models)
                if "CatBoosting Regressor" in model_name:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)
               
               
                y_train_pred = model.predict(X_train)

                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)

                test_model_score = r2_score(y_test, y_test_pred)

                report[model_name] = test_model_score
                logging.info(f"Model: {model_name}, Score: {test_model_score}")

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self,train_array,test_array):
        try:
           
            with open("yaml_config/params.yaml", "r") as f:
                config = yaml.safe_load(f)
            params = config['model_params']

            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(n_jobs=-1),
                "CatBoosting Regressor": CatBoostRegressor(task_type="CPU",thread_count=-1, verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            model_report:dict=self.evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            logging.info(model_report)
            print(model_report)
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.config.model_name,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)