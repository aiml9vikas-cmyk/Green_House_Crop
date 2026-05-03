# 🌱 Green House Crop Yield Prediction

An end-to-end Machine Learning pipeline to predict greenhouse crop yields ($kg/m^2$) based on environmental sensors, soil parameters, and crop varieties.

## 🚀 Features
- **Modular Architecture**: Clean separation of Ingestion, Validation, Transformation, Training, and Evaluation.
- **Automated Preprocessing**: Custom `FunctionTransformer` to calculate maturity days and handle date stripping automatically.
- **Experiment Tracking**: Integrated with **MLflow** and **Dagshub** for real-time metric logging.
- **Unified Master Pipeline**: Bundles preprocessing and the best model (XGBoost/CatBoost) into a single `.pkl` for zero-configuration inference.
- **Interactive UI**: Streamlit web application for manual entry and bulk CSV batch predictions.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **ML Frameworks**: Scikit-Learn, XGBoost, CatBoost
- **Tracking**: MLflow, Dagshub
- **Interface**: Streamlit
- **Environment**: Managed via `uv` or `venv`

## 📂 Project Structure
```text
Green_House_Crop/
├── artifacts/               # Generated data, preprocessors, and models
├── notebook/                # Research and EDA (.ipynb)
├── src/
│   └── Green_House_Crop/
│       ├── components/      # Core logic (DataIngestion, ModelTrainer, etc.)
│       ├── pipeline/        # Stage-wise training and prediction pipelines
│       ├── entity/          # Dataclass configurations
│       └── utils/           # Helper functions (YAML reader, object savers)
├── yaml_config/
│   ├── config.yaml          # Path configurations
│   ├── params.yaml          # Hyperparameters for GridSearchCV
│   └── schema.yaml          # Data validation rules
├── main.py                  # Pipeline orchestrator
└── app.py                   # Streamlit Web UI
```

## ⚙️ Installation & Usage

### 1. Setup Environment
```bash
# Using uv (Recommended)
uv sync
```

### 2. Run the Training Pipeline
This executes the full workflow from raw data to a registered model in Dagshub.
```bash
python -m main.py
    or
uv run main.py

```

### 3. Launch the Web Interface
```bash
streamlit run app.py
    or
uv run streamlit run app.py
```

## 📊 Model Performance
The current pipeline evaluates multiple models using **GridSearchCV** with **Early Stopping**. 
- **Leader**: XGBRegressor / CatBoost
- **Current R² Score**: ~0.81+
- **Metrics Tracked**: RMSE, MAE, R²

## 🧪 Future Improvements
- **GPU Support**: Enable ROCm/DirectML for Radeon GPU acceleration on larger datasets.
- **Feature Engineering**: Extracting seasonal "month" features from planting dates.
- **API Deployment**: Containerizing the `PredictPipeline` using Docker for AWS/Azure deployment.
