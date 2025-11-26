"""A collection of transformer blocks that combine FFN, Attn and normalization."""

import torch

from models.components.layers.attention import build_attention
from models.components.layers.feedforward import build_ffn
from models.components.layers.normalization import build_normalization


class GenericTransformerBlock(torch.nn.Module):
    """A simple transformer block that combines FFN, Attn and normalization.

    Args:
        hidden_dim (int): The hidden dimension of the model.
        context_window (int): The maximum sequence length.
        use_rope (bool): Whether to use Rotary Positional Embeddings.
        ffn_cfg (dict): Configuration for the Feed Forward Network.
        attn_cfg (dict): Configuration for the Attention mechanism.
    """

    def __init__(
        self,
        hidden_dim: int,
        context_window: int,
        use_rope: bool,
        ffn_cfg: dict,
        attn_cfg: dict,
    ):
        super().__init__()

        # build the attn norm
        self.attn_norm = build_normalization(
            normalization_name=attn_cfg["normalization"],
            dim=hidden_dim,
            bias=attn_cfg["bias"],
        )

        # build the attention
        self.attn = build_attention(
            hidden_dim=hidden_dim,
            context_window=context_window,
            use_rope=use_rope,
            attn_cfg=attn_cfg,
        )

        # build the ffn norm
        self.ffn_norm = build_normalization(
            normalization_name=ffn_cfg["normalization"],
            dim=hidden_dim,
            bias=ffn_cfg["bias"],
        )

        # build the ffn block
        self.ffn = build_ffn(
            hidden_dim=hidden_dim,
            ffn_cfg=ffn_cfg,
        )

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """A simple, residual forward pass through the GPT block.

        Args:
            x (torch.Tensor): The input tensor of shape (B, S, H).
            attention_mask (torch.Tensor, optional): The attention mask.

        Returns:
            torch.Tensor: The output tensor of shape (B, S, H).
        """
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
