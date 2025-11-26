"""The standard Model Shell.

It combines the embedding model, core model and LM head into a single object.
"""

import torch

from models.architectures import core as core_models
from models.embeddings import embedder as embedding_models
from models.heads import heads as model_heads


class ModelShell(torch.nn.Module):
    """Unify the embedding model, core model and LM head into a single object.

    Initializes the weights and prints basic model statistics.

    Args:
        embedding_model (embedding_models.EmbedderInterface): The embedding model.
        core_model (core_models.GenericTransformer): The core transformer model.
        model_head (model_heads.AutoregressiveLMHead): The language model head.
        weight_init_func (callable, optional): Function to initialize weights.
    """

    def __init__(
        self,
        embedding_model: embedding_models.EmbedderInterface,
        core_model: core_models.GenericTransformer,
        model_head: model_heads.AutoregressiveLMHead,
        weight_init_func=None,
    ):
        super().__init__()
        self.embedding_model = embedding_model
        self.core_model = core_model
        self.model_head = model_head

        # initialize model weights
        if weight_init_func is not None:
            self.apply(weight_init_func)
        self.device = torch.device("cpu")

    # override to device to set the attribute
    def to(self, *args, **kwargs):
        """Move the model to the specified device."""
        self.device = args[0]
        return super().to(*args, **kwargs)

    def forward(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The default forward pass is used for training.

        Args:
            token_ids (torch.Tensor): Input token IDs of shape (B, S).
            attention_mask (torch.Tensor, optional): Mask of shape (B, S) where 1=attend, 0=ignore padding.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - output: Model logits of shape (B, S, V).
                - aux_loss: Auxiliary loss (if any).
        """
        # pass the token_ids through the embedding model
        # to get B, S, H (with pos encoding if necessary)
        x = self.embedding_model(token_ids)

        # pass the embeddings through the core model with attention mask
        x = self.core_model(x, attention_mask=attention_mask)

        # pass the core model output through the model head
        x = self.model_head(x)

        return x

    @torch.no_grad()
    def inference(self, model_input) -> tuple[torch.Tensor, torch.Tensor]:
        """Takes a string or list of token ids as input, and returns the decoded model output.

        The actual decoding should happen in the decoding generator.

        Args:
            model_input (str or torch.Tensor): Input string or tensor of token IDs.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - logits: Final token logits of shape (B, V).
                - token_ids: The input token IDs used.
        """
        # check if input is string
        if isinstance(model_input, str):
            # use inference function of the embedding model
            token_ids_list = self.embedding_model.tokenize_input(
                model_input, truncate=True, add_eot=False
            )
            token_ids = torch.tensor(
                token_ids_list, device=self.device, dtype=torch.long
            ).unsqueeze(0)
        elif isinstance(model_input, list):
            token_ids = torch.tensor(
                model_input, device=self.device, dtype=torch.long
            ).unsqueeze(0)
        else:
            # Assume tensor
            token_ids = model_input.to(self.device)
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)

        # pass the embeddings through the embedding model
        x = self.embedding_model(token_ids)

        # pass the embeddings through the core model
        x = self.core_model(x)

        # pass the core model output through the model head
        logits = self.model_head.inference(x)

        return logits, token_ids

    @torch.no_grad()
    def loglikelihood(
        self, prefixes: list[str], continuations: list[str]
    ) -> torch.Tensor:
        """Compute the loglikelihood of continuation tokens given a prefix.

        Args:
            prefixes (list[str]): List of prefix strings.
            continuations (list[str]): List of continuation strings.

        Returns:
            torch.Tensor: Log-likelihoods of shape (B).
        """
        total_strings = [
            f"{prefix} {cont}" for prefix, cont in zip(prefixes, continuations)
        ]
        input_tokens = [
            self.embedding_model.tokenize_input(string, truncate=True)
            for string in total_strings
        ]
        padded_batch, mask = self.embedding_model.pad_batch(
            input_tokens, direction="right"
        )

        input_tensor = padded_batch.detach().clone().to(self.device).long()
        logits, _ = self.forward(input_tensor)
        logits = logits[:, :-1].reshape(-1, logits.size(-1))
        target_tensor = input_tensor[:, 1:].reshape(-1)
        ll = torch.nn.functional.cross_entropy(logits, target_tensor, reduction="none")
        mask = mask[:, 1:].reshape(-1).to(ll.device)
        ll = ll * mask
        ll = ll.view(input_tensor.size(0), -1).sum(dim=1)
        return -ll
