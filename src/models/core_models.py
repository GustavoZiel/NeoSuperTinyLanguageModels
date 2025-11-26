"""Simple, flexible core models."""

import torch

from models.components.layers.transformer_blocks import GenericTransformerBlock


class GenericTransformer(torch.nn.Module):
    """Generic Transformer Class intended to be used for as
    broad a range of transformer models as possible.

    Args:
        model_cfg (dict): Model configuration dictionary.
    """

    def __init__(self, model_cfg: dict):
        super().__init__()

        # build the transformer
        self.transformer = torch.nn.ModuleDict(
            {
                "drop": torch.nn.Dropout(),
                "h": torch.nn.ModuleList(
                    [
                        GenericTransformerBlock(
                            hidden_dim=model_cfg["hidden_dim"],
                            context_window=model_cfg["context_window"],
                            use_rope=model_cfg["positional_encoding_type"] == "rope",
                            ffn_cfg=model_cfg["core_model"]["ffn"],
                            attn_cfg=model_cfg["core_model"]["attn"],
                        )
                        for _ in range(model_cfg["core_model"]["num_layers"])
                    ]
                ),
            }
        )

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Pass an input through the model with optional attention mask.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, H).
            attention_mask (torch.Tensor, optional): Attention mask where 1=attend, 0=ignore.

        Returns:
            torch.Tensor: Output tensor of shape (B, S, H).
        """
        # apply dropout
        x = self.transformer.drop(x)

        # pass through the transformer blocks with attention mask
        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)

        return x


class GenericFFNSharedTransfomer(GenericTransformer):
    """Generic Transformer Class that shares the weights
    between all FFN blocks (similar to https://arxiv.org/abs/2402.16840).

    Args:
        model_cfg (dict): Model configuration dictionary.
    """

    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg=model_cfg)

        # share the weights between transformer blocks
        ffn_0 = self.transformer.h[0].ffn

        for i in range(1, len(self.transformer.h)):
            # find all linear layers in the ffn subnets and tie them to the first layer
            for name, module in ffn_0.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_module = dict(self.transformer.h[i].ffn.named_modules())[
                        name
                    ]
                    target_module.weight = module.weight
                    target_module.bias = module.bias
