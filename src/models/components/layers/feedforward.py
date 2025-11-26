"""A collection of FFN blocks."""

import torch
import torch.nn.functional as F

from models.components.layers.activations import build_activation


class GenericFFN(torch.nn.Module):
    """A simple feedforward network.

    Args:
        hidden_dim (int): Hidden dimension.
        ffn_dim (int): FFN intermediate dimension.
        bias (bool): Whether to use bias.
        ffn_activation (str): Activation function name.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        bias: bool,
        ffn_activation: str,
    ):
        super().__init__()
        # build the ffn block
        self.linear_1 = torch.nn.Linear(hidden_dim, ffn_dim, bias=bias)

        self.activation = build_activation(activation_name=ffn_activation)

        self.linear_2 = torch.nn.Linear(ffn_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A simple forward pass through the FFN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x


class SwiGLUFFN(torch.nn.Module):
    """SwiGLU FFN.

    Implementation based on:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    originally from https://arxiv.org/abs/2002.05202

    N.B. does not support dropout.

    Args:
        hidden_dim (int): Hidden dimension.
        ffn_dim (int): FFN intermediate dimension.
        bias (bool): Whether to use bias.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        bias: bool,
    ):
        super().__init__()
        # build the linear functions
        self.linear_1 = torch.nn.Linear(hidden_dim, ffn_dim, bias=bias)

        self.linear_2 = torch.nn.Linear(ffn_dim, hidden_dim, bias=bias)

        self.linear_3 = torch.nn.Linear(hidden_dim, ffn_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A simple forward pass through the FFN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.linear_2(F.silu(self.linear_1(x)) * self.linear_3(x))


FFN_REGISTRY = {
    "generic": lambda hidden_dim, ffn_cfg: GenericFFN(
        hidden_dim=hidden_dim,
        ffn_dim=ffn_cfg["ffn_dim"],
        bias=ffn_cfg["bias"],
        ffn_activation=ffn_cfg["activation"],
    ),
    "swiglu": lambda hidden_dim, ffn_cfg: SwiGLUFFN(
        hidden_dim=hidden_dim,
        ffn_dim=ffn_cfg["ffn_dim"],
        bias=ffn_cfg["bias"],
    ),
}


def build_ffn(hidden_dim: int, ffn_cfg: dict) -> torch.nn.Module:
    """Build a feedforward network.

    Args:
        hidden_dim (int): Hidden dimension.
        ffn_cfg (dict): FFN configuration.

    Returns:
        torch.nn.Module: FFN module.
    """
    return FFN_REGISTRY[ffn_cfg["ffn_type"]](hidden_dim=hidden_dim, ffn_cfg=ffn_cfg)
