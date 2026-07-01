from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline

STAGES = [
    ("Data Ingestion", DataIngestionTrainingPipeline),
    ("Prepare Base Model", PrepareBaseModelTrainingPipeline),
    ("Training", ModelTrainingPipeline),
    ("Evaluation", EvaluationPipeline),
]

for name, Pipeline in STAGES:
    try:
        logger.info(f">>>>>> stage {name} started <<<<<<")
        Pipeline().main()
        logger.info(f">>>>>> stage {name} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
