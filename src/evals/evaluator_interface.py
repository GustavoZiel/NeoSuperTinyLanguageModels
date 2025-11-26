"""Defines the EvaluatorInterface class."""


class EvaluationInterface:
    """Interface for evaluating a model.

    Args:
        model: The model to evaluate.
    """

    def __init__(self, model):
        self.model = model

    def evaluate(self) -> dict:
        """Evaluate the model performance on a list of benchmarks.

        Returns:
            dict: A dictionary containing evaluation results.
        """
        raise NotImplementedError()
