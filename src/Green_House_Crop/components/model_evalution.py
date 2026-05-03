import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import pickle
from src.Green_House_Crop.entity.config_entity import ModelEvaluationConfig
from src.Green_House_Crop.constants import *
from src.Green_House_Crop.utils.common import read_yaml, create_directories,save_json
from pathlib import Path
import os
#os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/aiml9vikas-cmyk/Green_House_Crop.mlflow"
#os.environ["MLFLOW_TRACKING_USERNAME"]="vikas"
#os.environ["MLFLOW_TRACKING_PASSWORD"]="410f2620a6c712cc25fbc18c13b188de3c44f1f2"

import dagshub
dagshub.init(repo_owner='aiml9vikas-cmyk', repo_name='Green_House_Crop', mlflow=True)

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def log_into_mlflow(self):
        # 1. Load Data
        test_data = pd.read_csv(self.config.test_data_path)
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # 2. Load Preprocessor and Model using Pickle
        # Update your config_entity to have separate paths if they are separate files
        with open(self.config.pre_processing, 'rb') as f:
            preprocessor = pickle.load(f)
            
        with open(self.config.model_path, 'rb') as f:
            model = pickle.load(f)

        # 3. Transform features (Converts "Lettuce" to numeric)
        test_x_transformed = preprocessor.transform(test_x)

        # 4. MLflow Logging
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # USE THE TRANSFORMED DATA HERE
            predicted_qualities = model.predict(test_x_transformed)

            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            
            # Save metrics locally
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="Green_House_Crop_Model")
            else:
                mlflow.sklearn.log_model(model, "model")