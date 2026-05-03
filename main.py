from src.Green_House_Crop.exception import CustomException
from src.Green_House_Crop.logger import logging
from src.Green_House_Crop.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.Green_House_Crop.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.Green_House_Crop.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.Green_House_Crop.pipeline.model_training_pipeline import ModelTrainerTrainingPipeline
from src.Green_House_Crop.pipeline.model_evalution_pipeline import ModelEvaluationTrainingPipeline
import sys

STAGE_NAME = "Data Ingestion stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.initiate_data_ingestion_p()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise CustomException(e,sys)


STAGE_NAME = "Data Validation stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.initiate_data_validation_p()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise CustomException(e,sys)


STAGE_NAME = "Data Transformation stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_transformation = DataTransformationTrainingPipeline()
   train_arr,test_arr,pre_processing=data_transformation.initiate_data_transformation_p()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise CustomException(e,sys)

STAGE_NAME = "Model Trainer stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_training = ModelTrainerTrainingPipeline()
   data_training.initiate_model_training_p(train_arr,test_arr)
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise CustomException(e,sys)
"""
STAGE_NAME = "Model evaluation stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = ModelEvaluationTrainingPipeline()
   data_ingestion.initiate_model_evaluation_p()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise e

"""