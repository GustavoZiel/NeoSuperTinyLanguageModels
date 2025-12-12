"""A collection of positional encoding modules."""

import math

import torch


class LearnedPosEncoding(torch.nn.Module):
    """Basic learned positional encoding.

    Args:
        hidden_dim (int): The hidden dimension of the model.
        context_window (int): The maximum sequence length.
    """

    def __init__(self, hidden_dim: int, context_window: int):
        super().__init__()
        self.pe = torch.nn.Embedding(
            num_embeddings=context_window, embedding_dim=hidden_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Takes the input tensor and returns it positionally encoded.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, H).

        Returns:
            torch.Tensor: Positionally encoded tensor of shape (B, S, H).
        """
        if len(x.shape) >= 2:
            return x + (self.pe(torch.arange(x.size(1), device=x.device)).unsqueeze(0))
        else:
            return x + self.pe(torch.arange(x.size(1), device=x.device))


class IdentityEncoding(torch.nn.Module):
    """Identity encoding. Used when no positional encoding is needed (e.g. RoPE)."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the input tensor as is.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The input tensor.
        """
        return x


class SinCosPosEncoding(torch.nn.Module):
    """Sinusoidal positional encoding.

    Based on "Attention Is All You Need" (Vaswani et al., 2017).
    Implementation adapted from PyTorch examples.

    Args:
        hidden_dim (int): The hidden dimension of the model.
        context_window (int): The maximum sequence length.
    """

    def __init__(self, hidden_dim: int, context_window: int):
        super().__init__()
        pe = torch.zeros(context_window, hidden_dim)
        position = torch.arange(0, context_window, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # pe has shape (1, S, H)

        self.pe = torch.nn.Parameter(pe)  # hack for distributed data parallel
        self.pe.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, H).

        Returns:
            torch.Tensor: Tensor with positional encoding added.
        """
        return x + self.pe[:, : x.size(1)]


POSITIONAL_ENCODING_REGISTRY = {
    "learned": lambda dim, size, **_: LearnedPosEncoding(
        hidden_dim=dim, context_window=size
    ),
    "rope": lambda **_: IdentityEncoding(),
    "none": lambda **_: IdentityEncoding(),
    "sincos": lambda dim, size, **_: SinCosPosEncoding(
        hidden_dim=dim, context_window=size
    ),
}


def build_positional_encodings(model_cfg: dict) -> torch.nn.Module:
    """Given the positional encoding config, build it.

    Args:
        model_cfg (dict): The model configuration dictionary.
                          Must contain 'positional_encoding_type', 'hidden_dim', and 'context_window'.

    Returns:
        torch.nn.Module: The positional encoding module.
    """
    return POSITIONAL_ENCODING_REGISTRY[model_cfg["positional_encoding_type"]](
        dim=model_cfg["hidden_dim"], size=model_cfg["context_window"]
    )
