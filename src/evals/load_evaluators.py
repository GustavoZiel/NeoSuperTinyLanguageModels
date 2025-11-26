"""Given an evaluator name, load the evaluator."""

from evals.evaluator_interface import EvaluationInterface
from evals.finetuning.glue import FinetuningEvaluator
from evals.finetuning.qa import FinetuningQA
from evals.mcqs.benchmarks.stories_progression import ProgressionEvaluator
from evals.mcqs.mcq_evaluator import MCQEvaluator

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
