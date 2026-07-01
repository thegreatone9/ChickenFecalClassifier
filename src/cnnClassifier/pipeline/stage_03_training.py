from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_callbacks import PrepareCallback
from cnnClassifier.components.training import Training


class ModelTrainingPipeline:
    def main(self):
        config = ConfigurationManager()
        callback_list = PrepareCallback(config=config.get_prepare_callback_config()).get_tb_ckpt_callbacks()
        training = Training(config=config.get_training_config())
        training.get_base_model()
        training.train_valid_generator()
        training.train(callback_list=callback_list)
