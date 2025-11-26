"""A collection of metrics for evaluating models."""

import torch

from training.utils import aggregate_value


def accuracy_metric(confidences: torch.Tensor) -> float:
    """Calculate the accuracy of the model over a path_prob.

    Assume that the ground truth is the first element in the list.

    Args:
        confidences (torch.Tensor): (B, N) tensor of confidences.

    Returns:
        float: Accuracy.
    """
    _, predicted = torch.max(confidences, 1)
    ## aggregate the tensor values
    return aggregate_value((predicted == 0).float().mean())


def path_confidence(confidences: torch.Tensor) -> float:
    """Calculate the path confidence of the model.

    Assume that the ground truth is the first element in the list.

    Args:
        confidences (torch.Tensor): (B, N) tensor of confidences.

    Returns:
        float: Path confidence.
    """
    softmaxed = torch.nn.functional.softmax(confidences, dim=-1)
    softmaxed = softmaxed[:, 0]
    ## aggregate the tensor values
    return aggregate_value(softmaxed.mean())


def ground_confidence(confidences: torch.Tensor) -> float:
    """Calculate the confidence of the model on the ground truth.

    Assume that the ground truth is the first element in the list.

    Args:
        confidences (torch.Tensor): (B, N) tensor of confidences.

    Returns:
        float: Confidence on ground truth.

    See: https://arxiv.org/pdf/2406.04391 - this is equivalent to
    $$P_\\theta^{\\text{Choices}}(\\text{Ground Truth})$$ over the
    Path probabilities. (takeaway 3)
    """
    return confidences[:, 0].mean()


MCQ_METRIC_REGISTRY = {
    "accuracy": accuracy_metric,
    "path_confidence": path_confidence,
    "ground_confidence": ground_confidence,
}
