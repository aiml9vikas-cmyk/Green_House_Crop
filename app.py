import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from src.Green_House_Crop.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.Green_House_Crop.exception import CustomException

# Typical growth cycle in days mapping
crop_duration_map = {
    "Lettuce": 30,
    "Cucumber": 55,
    "Tomato": 75,
    "Pepper": 80
}

# Agricultural safety threshold mappings (Max allowed temperature threshold per crop)
crop_max_temp_limits = {
    "Lettuce": 28.0,
    "Cucumber": 38.0,
    "Tomato": 35.0,
    "Pepper": 40.0
}

# Expected industry standard yield references (min, typical, max benchmarks for the manual graph)
CROP_YIELD_BENCHMARKS = {
    "Lettuce": {"min": 5.0, "avg": 12.0, "max": 20.0},
    "Cucumber": {"min": 10.0, "avg": 22.0, "max": 35.0},
    "Tomato": {"min": 12.0, "avg": 25.0, "max": 45.0},
    "Pepper": {"min": 8.0, "avg": 18.0, "max": 30.0}
}

# Expected schema for CSV upload verification
REQUIRED_CSV_COLUMNS = [
    "greenhouse_id", "crop_type", "variety", "days_to_maturity", 
    "avg_temperature_C", "min_temperature_C", "max_temperature_C", 
    "humidity_percent", "co2_ppm", "light_intensity_lux", 
    "photoperiod_hours", "irrigation_mm", "fertilizer_N_kg_ha", 
    "fertilizer_P_kg_ha", "fertilizer_K_kg_ha", "pest_severity", 
    "soil_pH", "planting_date", "harvest_date"
]

# Page configuration
st.set_page_config(page_title="Greenhouse Yield Predictor", layout="wide")

st.title("🌱 Greenhouse Crop Yield Prediction")
st.markdown("Predict crop yield ($kg/m^2$) using manual entry or bulk CSV upload.")

# Toggle between Manual and CSV
input_mode = st.radio("Select Input Method:", ["Manual Entry", "Upload CSV for Batch Prediction"], horizontal=True)

if input_mode == "Manual Entry":
    is_valid_input = True

    # =========================================================================
    # STEP 1: CROP & TIMELINE SELECTION (Reactive - Outside Form)
    # =========================================================================
    st.subheader("Step 1: Crop & Timeline Selection")
    col1_top, col2_top = st.columns(2)
    
    with col1_top:
        crop_type = st.selectbox("Crop Type", ["Tomato", "Cucumber", "Lettuce", "Pepper"])
        p_date = st.date_input("Planting Date")
            
        typical_days = crop_duration_map[crop_type]
        default_harvest_date = p_date + pd.Timedelta(days=typical_days)
        
        h_date = st.date_input(
            "Expected Harvest Date", 
            value=default_harvest_date, 
            min_value=p_date
        )

        calculated_days = int((h_date - p_date).days)
        
        min_allowed_days = int(typical_days * 0.7)
        max_allowed_days = int(typical_days * 1.5)
        
        if calculated_days < min_allowed_days:
            st.error(f"⚠️ Timeline Error: {calculated_days} days is too short. Typical {crop_type} development requires at least {min_allowed_days} days from planting.")
            is_valid_input = False
        elif calculated_days > max_allowed_days:
            st.error(f"⚠️ Timeline Error: {calculated_days} days is too long. A standard greenhouse {crop_type} cycle should finish within {max_allowed_days} days.")
            is_valid_input = False
        
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
        st.metric(label="📆 Days to Maturity", value=f"{calculated_days} days (Typical: {typical_days})")
        days_to_maturity = float(calculated_days)

    # =========================================================================
    # STEP 2: TEMPERATURE CONFIGURATION & VALIDATION (Reactive - Outside Form)
    # =========================================================================
    st.subheader("Step 2: Temperature Parameters")
    col_temp1, col_temp2 = st.columns(2)
    
    with col_temp1:
        min_temp = st.number_input("Min Temperature (°C)", min_value=5.0, max_value=35.0, value=18.0)
        max_temp = st.number_input("Max Temperature (°C)", min_value=15.0, max_value=45.0, value=28.0)
        
        if min_temp > max_temp:
            st.error("⚠️ Mathematical Error: Minimum temperature cannot be higher than maximum temperature.")
            is_valid_input = False
            
        max_allowed = crop_max_temp_limits[crop_type]
        if max_temp > max_allowed:
            st.error(f"🚨 Heat Stress Hazard: Maximum temperature of {max_temp}°C exceeds the physiological survival limits for {crop_type} ({max_allowed}°C).")
            is_valid_input = False

    with col_temp2:
        calculated_avg = (min_temp + max_temp) / 2.0
        st.metric(label="📊 Calculated Avg Temperature", value=f"{calculated_avg:.1f} °C")
        avg_temp = float(calculated_avg)

    # =========================================================================
    # STEP 3: SOIL & OPERATIONAL PARAMETERS (Inside Form)
    # =========================================================================
    st.subheader("Step 3: Soil & Operational Parameters")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            greenhouse_id = st.selectbox("Greenhouse ID", [1.0, 2.0, 3.0, 4.0, 5.0])
            co2 = st.number_input("CO2 level (ppm)", min_value=300.0, max_value=1500.0, value=800.0)
            soil_ph = st.number_input("Soil pH", min_value=4.0, max_value=9.0, value=6.5)

        with col2:
            photoperiod = st.number_input("Photoperiod (hours)", min_value=0.0, max_value=24.0, value=12.0)
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 75.0)
            irrigation = st.number_input("Irrigation (mm)", min_value=0.0, max_value=20.0, value=7.0)
            
        with col3:
            light = st.number_input("Light Intensity (lux)", min_value=5000.0, max_value=60000.0, value=30000.0)
            fert_n = st.number_input("Fertilizer N (kg/ha)", min_value=0.0, max_value=300.0, value=150.0)
            fert_p = st.number_input("Fertilizer P (kg/ha)", min_value=0.0, max_value=150.0, value=70.0)
            fert_k = st.number_input("Fertilizer K (kg/ha)", min_value=0.0, max_value=300.0, value=170.0)
            pest_severity = st.slider("Pest Severity (0-5)", 0.0, 5.0, 0.5)

        submit_button = st.form_submit_button("Predict Yield")

    # Evaluate execution gating conditions
    if submit_button:
        if not is_valid_input:
            st.warning("🛑 Pipeline Blocked: Please resolve the operational safety alerts highlighted above.")
        else:
            try:
                custom_data_instance = CustomData(
                    greenhouse_id=greenhouse_id, crop_type=crop_type, variety=variety,
                    days_to_maturity=days_to_maturity, avg_temperature_C=avg_temp,
                    min_temperature_C=min_temp, max_temperature_C=max_temp,
                    humidity_percent=humidity, co2_ppm=co2, light_intensity_lux=light,
                    photoperiod_hours=photoperiod, irrigation_mm=irrigation,
                    fertilizer_N_kg_ha=fert_n, fertilizer_P_kg_ha=fert_p,
                    fertilizer_K_kg_ha=fert_k, pest_severity=pest_severity, soil_pH=soil_ph,
                    planting_date=str(p_date), harvest_date=str(h_date)
                )
                
                pred_df = custom_data_instance.get_data_as_data_frame()
                predict_pipeline = PredictPipeline()
                results = predict_pipeline.predict(pred_df)
                yield_value = results.item() 
                
                st.success(f"🎉 Successful Prediction!")
                
                # Split output window into columns to seat Metric next to the manual chart
                m_col1, m_col2 = st.columns([1, 2])
                with m_col1:
                    st.metric(label="📊 Estimated Crop Yield", value=f"{yield_value:.2f} kg/m²")
                
                with m_col2:
                    # --- NEW PLOTLY MANUAL ENTRY CHART INTEGRATION ---
                    bench = CROP_YIELD_BENCHMARKS[crop_type]
                    fig_manual = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = yield_value,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"Yield Performance Range vs. Standard {crop_type}", 'font': {'size': 14}},
                        gauge = {
                            'axis': {'range': [0, bench["max"] * 1.2], 'tickwidth': 1},
                            'bar': {'color': "#2ca02c"},
                            'steps': [
                                {'range': [0, bench["min"]], 'color': '#ff9999'},
                                {'range': [bench["min"], bench["avg"]], 'color': '#ffffcc'},
                                {'range': [bench["avg"], bench["max"] * 1.2], 'color': '#c2f0c2'}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 3},
                                'thickness': 0.75,
                                'value': bench["avg"]}
                        }
                    ))
                    fig_manual.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_manual, use_container_width=True)
                
            except Exception as e:
                st.error("🚨 An error occurred in the machine learning pipeline.")
                st.exception(e)

# =========================================================================
# STEP 4: BATCH CSV FILE PROCESSING MODALITY
# =========================================================================
else:
    st.subheader("Bulk File Processing")
    uploaded_file = st.file_uploader("Upload Greenhouse Metric Sheet (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            missing_cols = [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing Layout Requirements. Missing headers: {missing_cols}")
            else:
                st.info("🔄 Processing CSV: Starting data handling and biological evaluation...")
                
                # --- AUTOMATED MISSING VALUE IMPUTATION ---
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                imputation_logs = []
                
                for col in numeric_cols:
                    if df[col].isnull().any():
                        missing_count = df[col].isnull().sum()
                        group_medians = df.groupby('crop_type')[col].transform('median')
                        fallback_median = df[col].median()
                        fill_values = group_medians.fillna(fallback_median)
                        
                        df[col] = df[col].fillna(fill_values)
                        imputation_logs.append(f"🔧 Imputed **{missing_count}** missing entries in `{col}` using crop-specific medians.")

                # --- BIOLOGICAL ANOMALY CLEANING & ALERT LOGGING ---
                alert_logs = []
                initial_row_count = len(df)
                
                temp_mismatch_mask = df['min_temperature_C'] > df['max_temperature_C']
                mismatch_indices = df[temp_mismatch_mask].index.tolist()
                if mismatch_indices:
                    alert_logs.append(f"❌ Dropped **{len(mismatch_indices)}** rows due to Mathematical Errors (`min_temperature_C` > `max_temperature_C`).")
                
                limit_thresholds = df['crop_type'].map(crop_max_temp_limits)
                heat_stress_mask = df['max_temperature_C'] > limit_thresholds
                heat_indices = df[heat_stress_mask].index.tolist()
                if heat_indices:
                    alert_logs.append(f"❌ Dropped **{len(heat_indices)}** rows due to Heat Stress Hazards (Exceeded physiological survival limits).")
                
                anomaly_mask = temp_mismatch_mask | heat_stress_mask
                cleaned_df = df[~anomaly_mask].copy()
                
                # --- DISPLAY CLEANING AND IMPUTATION SUMMARY ---
                with st.expander("📊 Data Cleaning & Imputation Logs", expanded=True):
                    if not imputation_logs and not alert_logs:
                        st.success("✅ Dataset structure is flawless! No missing values or biological anomalies detected.")
                    else:
                        if imputation_logs:
                            for log in imputation_logs:
                                st.markdown(log)
                        if alert_logs:
                            for log in alert_logs:
                                st.markdown(log)
                
                if len(cleaned_df) == 0:
                    st.error("🛑 Processing Aborted: Every row in the uploaded CSV failed environmental threshold sanity checks.")
                else:
                    predict_pipeline = PredictPipeline()
                    batch_results = predict_pipeline.predict(cleaned_df)
                    cleaned_df["Predicted_Yield_kg_m2"] = batch_results
                    
                    # --- PERFORMANCE CHART GENERATION (Plotly Batch Code) ---
                    st.subheader("📊 Visual Performance Analysis")
                    chart_df = cleaned_df.groupby(["greenhouse_id", "crop_type"], as_index=False)["Predicted_Yield_kg_m2"].mean()
                    
                    fig = px.bar(
                        chart_df,
                        x="greenhouse_id",
                        y="Predicted_Yield_kg_m2",
                        color="crop_type",
                        barmode="group",
                        title="Average Predicted Yield by Greenhouse Unit & Crop Variant",
                        labels={"greenhouse_id": "Greenhouse ID", "Predicted_Yield_kg_m2": "Avg Yield (kg/m²)", "crop_type": "Crop Selection"},
                        text_auto='.2f'
                    )
                    
                    fig.update_layout(xaxis_type='category')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("📋 Output Dataset View")
                    st.dataframe(cleaned_df)
                    
                    csv_download = cleaned_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Cleaned Predictions CSV",
                        data=csv_download,
                        file_name="greenhouse_yield_predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error("🚨 Failed to process the bulk data file.")
            st.exception(e)
