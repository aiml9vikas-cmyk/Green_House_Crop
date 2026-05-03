import streamlit as st
import pandas as pd
from src.Green_House_Crop.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.Green_House_Crop.exception import CustomException
# Page configuration
import sys
st.set_page_config(page_title="Greenhouse Yield Predictor", layout="wide")

st.title("🌱 Greenhouse Crop Yield Prediction")
st.markdown("Enter the greenhouse parameters below to predict the crop yield ($kg/m^2$).")
# 1. Place these ABOVE/OUTSIDE the form for dynamic behavior
col1_top, col2_top = st.columns(2)
with col1_top:
    crop_type = st.selectbox("Crop Type", ["Tomato", "Cucumber", "Lettuce", "Pepper"])

with col2_top:
    if crop_type == "Tomato":
        varieties = ["Heirloom", "Cherry", "Beefsteak", "Roma"]
    elif crop_type == "Cucumber":
        varieties = ["Slicing", "Pickling", "English"]
    elif crop_type == "Lettuce":
        varieties = ["Butterhead", "Leaf", "Iceberg", "Romaine"]
    else:  # Pepper
        varieties = ["Habanero", "Bell", "Jalapeno"]
    
    variety = st.selectbox("Variety", varieties)

# Create a form for user inputs to prevent constant reloading
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        greenhouse_id = st.selectbox("Greenhouse ID", [1.0, 2.0, 3.0, 4.0, 5.0])
        
        # CO2 in CSV ranges from 430 to 1172
        co2 = st.number_input("CO2 level (ppm)", min_value=300.0, max_value=1500.0, value=800.0)
        # Soil pH in your CSV ranges from 5.3 to 8.0
        soil_ph = st.number_input("Soil pH", min_value=4.0, max_value=9.0, value=6.5)
        photoperiod = st.number_input("Photoperiod (hours)", min_value=0.0, max_value=24.0, value=12.0)
        days_to_maturity = st.number_input("Days to Maturity", min_value=1.0, value=60.0)

    with col2:
        
        # Values based on CSV: Avg temp ranges from ~14 to 31
        avg_temp = st.number_input("Avg Temperature (°C)", min_value=10.0, max_value=40.0, value=22.0)
        
        # Min temp ranges from ~13 to 30
        min_temp = st.number_input("Min Temperature (°C)", min_value=5.0, max_value=35.0, value=18.0)
        
        # Max temp ranges from ~16 to 33
        max_temp = st.number_input("Max Temperature (°C)", min_value=15.0, max_value=45.0, value=28.0)
        
        # Humidity in CSV ranges from 61% to 91%
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 75.0)
         # Pest severity in your CSV goes up to 4.7; let's use a 0-5 scale
        pest_severity = st.slider("Pest Severity (0-5)", 0.0, 5.0, 0.5)
        
        
      
       

    with col3:
         # Irrigation ranges from ~3.6 to 11.8 in your data
        irrigation = st.number_input("Irrigation (mm)", min_value=0.0, max_value=20.0, value=7.0)
          # Light in CSV ranges from 10,000 to 54,000
        light = st.number_input("Light Intensity (lux)", min_value=5000.0, max_value=60000.0, value=30000.0)

        # Fertilizers (N, P, K) ranges based on your CSV data
        fert_n = st.number_input("Fertilizer N (kg/ha)", min_value=0.0, max_value=300.0, value=150.0)
        fert_p = st.number_input("Fertilizer P (kg/ha)", min_value=0.0, max_value=150.0, value=70.0)
        fert_k = st.number_input("Fertilizer K (kg/ha)", min_value=0.0, max_value=300.0, value=170.0)
        
       

    submit_button = st.form_submit_button("Predict Yield")

# Handling the prediction
if submit_button:
    try:
        # 1. Initialize CustomData with form values
        data = CustomData(
            greenhouse_id=greenhouse_id, crop_type=crop_type, variety=variety,
            days_to_maturity=days_to_maturity, avg_temperature_C=avg_temp,
            min_temperature_C=min_temp, max_temperature_C=max_temp,
            humidity_percent=humidity, co2_ppm=co2, light_intensity_lux=light,
            photoperiod_hours=photoperiod, irrigation_mm=irrigation,
            fertilizer_N_kg_ha=fert_n, fertilizer_P_kg_ha=fert_p,
            fertilizer_K_kg_ha=fert_k, pest_severity=pest_severity, soil_pH=soil_ph
        )

        # 2. Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        st.write("### Input Data Summary", pred_df)

        # 3. Run Prediction Pipeline
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # 4. Display Results
        st.success(f"### 📈 Predicted Yield: {results[0]:.2f} kg/m²")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        raise CustomException(e, sys)