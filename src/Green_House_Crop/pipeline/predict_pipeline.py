import sys
import pandas as pd
from src.Green_House_Crop.exception import CustomException
from src.Green_House_Crop.utils.common import load_object
import os
import pickle
import streamlit as st

# CRITICAL: Imports for pickle reconstruction
from src.Green_House_Crop.components.data_transformation import GrowthDurationTransformer, OutlierCapper

# 1. STANDALONE CACHED FUNCTION (Correct way to use st.cache_resource)
@st.cache_resource
def load_essentials():
    """Loads model and preprocessor once and keeps them in memory."""
    model_path = os.path.join("artifacts", "data_training", "model.pickel")
    preprocessor_path = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")
    
    model = load_object(file_path=model_path)
    preprocessor = load_object(file_path=preprocessor_path)
    return model, preprocessor

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # 2. Retrieve cached objects instantly
            model, preprocessor = load_essentials()
            
            # 3. Transform and Predict
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, **kwargs):
        self.data_dict = {k: [v] for k, v in kwargs.items()}
        
    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame(self.data_dict)
        except Exception as e:
            raise CustomException(e, sys)