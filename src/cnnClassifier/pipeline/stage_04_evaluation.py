from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.evaluation import Evaluation


class EvaluationPipeline:
    def main(self):
        config = ConfigurationManager()
        evaluation = Evaluation(config.get_validation_config())
        evaluation.evaluation()
        evaluation.save_score()