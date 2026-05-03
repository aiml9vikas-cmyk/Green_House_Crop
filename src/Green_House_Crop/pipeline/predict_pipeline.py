import sys
import pandas as pd
from src.Green_House_Crop.exception import CustomException
from src.Green_House_Crop.utils.common import load_object
import os
import pickle
import streamlit as st

@st.cache_resource
class PredictPipeline:
    def __init__(self):
        pass
    
  
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts", "data_training", "model.pickel")
            preprocessor_path = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")
            
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        greenhouse_id: float,
        crop_type: str,
        variety: str,
        days_to_maturity: float,
        avg_temperature_C: float,
        min_temperature_C: float,
        max_temperature_C: float,
        humidity_percent: float,
        co2_ppm: float,
        light_intensity_lux: float,
        photoperiod_hours: float,
        irrigation_mm: float,
        fertilizer_N_kg_ha: float,
        fertilizer_P_kg_ha: float,
        fertilizer_K_kg_ha: float,
        pest_severity: float,
        soil_pH: float):

        self.data_dict = {
            "greenhouse_id": [greenhouse_id],
            "crop_type": [crop_type],
            "variety": [variety],
            "days_to_maturity": [days_to_maturity],
            "avg_temperature_C": [avg_temperature_C],
            "min_temperature_C": [min_temperature_C],
            "max_temperature_C": [max_temperature_C],
            "humidity_percent": [humidity_percent],
            "co2_ppm": [co2_ppm],
            "light_intensity_lux": [light_intensity_lux],
            "photoperiod_hours": [photoperiod_hours],
            "irrigation_mm": [irrigation_mm],
            "fertilizer_N_kg_ha": [fertilizer_N_kg_ha],
            "fertilizer_P_kg_ha": [fertilizer_P_kg_ha],
            "fertilizer_K_kg_ha": [fertilizer_K_kg_ha],
            "pest_severity": [pest_severity],
            "soil_pH": [soil_pH]
        }
        
    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame(self.data_dict)
        except Exception as e:
            raise CustomException(e, sys)
