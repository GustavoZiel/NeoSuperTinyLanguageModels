"""Integration code for model evaluation."""

import torch

from models import generator
from models import shell as model_shell


def batch(data: list, batch_size: int):
    """Yield batches from a list.

    Args:
        data (list): The list to batch.
        batch_size (int): The size of each batch.

    Yields:
        list: A batch of data.
    """
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


class EvalWrapper:
    """Wrapper class for evaluating a model shell.

    Args:
        model_shell (model_shell.ModelShell): The model shell to evaluate.
    """

    def __init__(self, model_shell: model_shell.ModelShell):
        self.model_shell = model_shell

    def loglikelihood(
        self, prefixes: list[str], continuations: list[str]
    ) -> list[float]:
        """Compute the loglikelihood of given inputs.

        Args:
            prefixes (list[str]): List of prefix strings.
            continuations (list[str]): List of continuation strings.

        Returns:
            list[float]: List of loglikelihood scores.
        """
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        self.model_shell = self.model_shell.to(device)
        results = []
        with torch.no_grad():
            with torch.autocast(device_type=device_str):
                for prefix_batch, cont_batch in zip(
                    batch(prefixes, 32), batch(continuations, 32)
                ):
                    ll = self.model_shell.loglikelihood(prefix_batch, cont_batch)
                    results.extend(ll.cpu().numpy())
        return results

    def generate(self, prefixes: list[str]) -> list[str]:
        """Generate a continuation for a given prefix.

        Args:
            prefixes (list[str]): List of prefix strings.

        Yields:
            str: Generated continuation.
        """
        model_generator = generator.StandardGenerator(
            self.model_shell,
            generate_cfg={
                "max_new_tokens": 128,
                "temperature": 0.7,
                "top_k": 0.9,
            },
        )
        for prefix in prefixes:
            # tokenize the inputs
            yield model_generator.default_generate(prefix)
