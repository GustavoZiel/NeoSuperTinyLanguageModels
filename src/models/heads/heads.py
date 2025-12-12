"""A collection of different model heads."""

import torch

from models.layers.normalization import build_normalization


class AutoregressiveLMHead(torch.nn.Module):
    """Generic autoregressive language model head.

    Args:
        model_cfg (dict): Model configuration dictionary.
    """

    def __init__(self, model_cfg: dict):
        super().__init__()
        self.layer_norm = build_normalization(
            normalization_name=model_cfg["lm_head"]["normalization"],
            dim=model_cfg["hidden_dim"],
            bias=model_cfg["lm_head"]["bias"],
        )
        self.linear = torch.nn.Linear(
            in_features=model_cfg["hidden_dim"],
            out_features=model_cfg["vocab_size"],
            bias=model_cfg["lm_head"]["bias"],
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Pass the input through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, H).

        Returns:
            tuple[torch.Tensor, None]:
                - Output tensor of shape (B, S, V).
                - None (placeholder for auxiliary loss).
        """
        # apply layer norm
        x = self.layer_norm(x)

        # pass through the linear layer
        x = self.linear(x)

        return x, None

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Pass the input through the model, then return the final token logits.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, H).

        Returns:
            torch.Tensor: Final token logits of shape (B, V).
        """
        return self.forward(x)[0][:, -1, :]
