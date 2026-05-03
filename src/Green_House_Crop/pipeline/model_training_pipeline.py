from src.Green_House_Crop.config.configuration import ConfigurationManager
from src.Green_House_Crop.components.model_training import ModelTrainer
from src.Green_House_Crop.components.data_transformation import DataTransformation
from src.Green_House_Crop.exception import CustomException
from src.Green_House_Crop.logger import logging
import sys

STAGE_NAME = "Model Trainer stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training_p(self,train_arr,test_arr):
        try:
            config = ConfigurationManager()
            # 2. Run Model Training
            # Pass the arrays directly to the trainer
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_score = model_trainer.initiate_model_trainer(
                train_array=train_arr, 
                test_array=test_arr
            )
            
            print(f"Model Training Complete. Best Score: {model_score}")

        except Exception as e:
            raise CustomException(e, sys)