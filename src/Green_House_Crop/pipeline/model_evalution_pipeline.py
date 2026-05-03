from src.Green_House_Crop.config.configuration import ConfigurationManager
from src.Green_House_Crop.components.model_evalution import ModelEvaluation
from src.Green_House_Crop.logger import logging

STAGE_NAME = "Model evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation_p(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.log_into_mlflow()