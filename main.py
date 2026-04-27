from src.Green_House_Crop.logger import logging
from src.Green_House_Crop.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.Green_House_Crop.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
"""
from src.Green_House_Crop.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.Green_House_Crop.pipeline.model_trainer_pipeline import ModelTrainerTrainingPipeline
from src.Green_House_Crop.pipeline.model_evaluation_pipeline import ModelEvaluationTrainingPipeline
"""
STAGE_NAME = "Data Ingestion stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.initiate_data_ingestion_p()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Data Validation stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.initiate_data_validation_p()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
