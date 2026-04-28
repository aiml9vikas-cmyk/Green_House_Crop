from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    raw_data_path: Path


@dataclass
class DataValidationConfig:
    root_dir:Path
    STATUS_FILE:str
    raw_data_path:Path
    all_schema:dict

@dataclass
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    raw_data_path: Path
    pre_processing: Path
    target_column: float
    all_schema: dict

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str