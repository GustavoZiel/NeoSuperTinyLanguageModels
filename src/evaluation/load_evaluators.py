"""Given an evaluator name, load the evaluator."""

from evaluation.evaluator_interface import EvaluationInterface
from evaluation.finetuning.glue import FinetuningEvaluator
from evaluation.finetuning.qa import FinetuningQA
from evaluation.mcqs.benchmarks.stories_progression import ProgressionEvaluator
from evaluation.mcqs.mcq_evaluator import MCQEvaluator

EVALUATOR_REGISTRY = {
    "mcq": MCQEvaluator,
    "glue": FinetuningEvaluator,
    "ft_qa": FinetuningQA,
    "prog": ProgressionEvaluator,
}


def load_evaluator(evaluator_name: str, model, **kwargs) -> EvaluationInterface:
    """Given the evaluator name, load the evaluator.

    Args:
        evaluator_name (str): The name of the evaluator to load.
        model: The model to evaluate.
        **kwargs: Additional arguments passed to the evaluator constructor.

    Returns:
        EvaluationInterface: The instantiated evaluator.
    """
    return EVALUATOR_REGISTRY[evaluator_name](model, **kwargs)
