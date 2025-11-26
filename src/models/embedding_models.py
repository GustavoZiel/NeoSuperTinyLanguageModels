"""A collection of embedding models.

A collection model includes the tokenizer(s), token embeddings,
and positional encodings (if necessary).
"""

import torch

from models.components.positional_encoding import build_positional_encodings
from models.components.tokenizers import build_tokenizer


class EmbedderInterface(torch.nn.Module):
    """Interface for the embedder component of the model."""

    def __init__(self):
        super().__init__()
        self.eot_token = None  # Should be set by subclasses

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """Takes the token_ids as input and returns the embeddings.

        Args:
            token_ids (torch.LongTensor): Input token IDs.

        Returns:
            torch.Tensor: Embeddings.
        """
        raise NotImplementedError

    def tokenize_input(
        self, input_string: str, truncate: bool = False, add_eot: bool = True
    ) -> list[int]:
        """Takes a single input string and returns the tokenized input.

        Args:
            input_string (str): The input string.
            truncate (bool): Whether to perform (left) truncation.
            add_eot (bool): Whether to add the End-Of-Text token.

        Returns:
            list[int]: Token IDs.
        """
        raise NotImplementedError

    def decode(self, tokens: torch.LongTensor) -> list[str]:
        """Decodes a tensor of tokens into a list of strings.

        Args:
            tokens (torch.LongTensor): Tensor of tokens of shape (B, S).

        Returns:
            list[str]: List of decoded strings.
        """
        raise NotImplementedError

    def inference(self, input_string: str, add_eot: bool = False) -> torch.Tensor:
        """Maps string to embeddings for inference.

        Args:
            input_string (str): The input string.
            add_eot (bool): Whether to add the End-Of-Text token.

        Returns:
            torch.Tensor: Embeddings of shape (1, S, H).
        """
        token_ids = self.tokenize_input(input_string, truncate=True, add_eot=add_eot)
        token_ids = (
            torch.tensor(token_ids).unsqueeze(0).to(next(self.parameters()).device)
        )
        return self.forward(token_ids)

    def pad_batch(
        self, token_lists: list[list[int]], direction: str = "right"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad a list of token lists to the same length.

        Args:
            token_lists (list[list[int]]): List of lists of tokens.
            direction (str): Padding direction ("left" or "right").

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Padded tensor and mask tensor.
        """
        raise NotImplementedError

    def truncate(self, token_lists: list[list[int]]) -> list[list[int]]:
        """Truncate a list of token lists to be shorter than the maximum length.

        Args:
            token_lists (list[list[int]]): List of lists of tokens.

        Returns:
            list[list[int]]: Truncated token lists.
        """
        raise NotImplementedError

    def get_sequence_info(self, x: torch.Tensor) -> tuple[list[int], torch.Tensor]:
        """Given a batch of sequences of tokens, return the character lengths and mask.

        Args:
            x (torch.Tensor): Batch of token sequences (B, S).

        Returns:
            tuple[list[int], torch.Tensor]:
                - sequence_char_lengths: List of character lengths for each sequence.
                - mask: Boolean mask where True indicates valid tokens (not pad/EOT).
        """
        sequence_char_lengths = []
        # then we decode everything
        # batch decode
        sequences = self.tokenizer.decode_batch(x)
        for seq in sequences:
            sequence_char_lengths.append(len(seq))

        # obtain the mask for end-of-word and pad tokens
        mask = x != self.tokenizer.pad_token
        mask = mask & (x != self.tokenizer.eot_token)

        return (
            sequence_char_lengths,
            mask,
        )


class GenericEmbedder(EmbedderInterface):
    """A simple and flexible embedding model.

    All embedders should inherit from this class.

    Args:
        model_cfg (dict): Model configuration dictionary.
    """

    def __init__(self, model_cfg: dict):
        super().__init__()
        # build the tokenizer
        self.tokenizer = build_tokenizer(
            tokenizer_type=model_cfg["embedder"]["tokenizer_type"],
            vocab_size=model_cfg["vocab_size"],
            dataset_name=model_cfg["embedder"]["dataset_name"],
        )

        # build the token embeddings
        self.token_embedder = torch.nn.Embedding(
            num_embeddings=model_cfg["vocab_size"],
            embedding_dim=model_cfg["hidden_dim"],
        )

        # build the positional encodings
        self.positional_encodings = build_positional_encodings(model_cfg=model_cfg)
        self.eot_token = self.tokenizer.eot_token
        self.model_cfg = model_cfg

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Takes the token_ids as input and returns the embeddings.

        To obtain the token ids, use `.tokenize_input()`.

        Args:
            token_ids (torch.Tensor): Input token IDs of shape (B, S).

        Returns:
            torch.Tensor: Embeddings of shape (B, S, H).
        """
        # get the token embeddings
        x = self.token_embedder(token_ids)

        # apply the positional encoding, if any
        x = self.positional_encodings(x)

        return x

    def tokenize_input(
        self, input_string: str, truncate: bool = False, add_eot: bool = True
    ) -> list[int]:
        """Tokenizes the given input string into a list of token IDs.

        Args:
            input_string (str): The input string to tokenize.
            truncate (bool, optional): If True, truncates the tokenized sequence according to model constraints. Defaults to False.
            add_eot (bool, optional): If True, appends the end-of-text (EOT) token to the tokenized sequence. Defaults to True.

        Returns:
            list[int]: A list of token IDs representing the tokenized input string.
        """
        token_ids = self.tokenizer.encode(input_string)
        if add_eot:
            token_ids.append(self.eot_token)
        if truncate:
            token_ids = self.truncate([token_ids])[0]
        return token_ids

    def pad_batch(
        self, token_lists: list[list[int]], direction: str = "right"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad a list of token lists to the same length.

        Args:
            token_lists (list[list[int]]): List of lists of tokens.
            direction (str): Padding direction ("left" or "right").

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Padded tensor and mask tensor.
        """
        return self.tokenizer.pad_batch(token_lists, direction=direction)

    def truncate(self, token_lists: list[list[int]]) -> list[list[int]]:
        """Truncate a list of token lists to be shorter than the maximum length.

        Args:
            token_lists (list[list[int]]): List of lists of tokens.

        Returns:
            list[list[int]]: Truncated token lists.
        """
        # get model max length
        max_length = self.model_cfg["context_window"]
        return [token_seq[-max_length:] for token_seq in token_lists]

    def decode(self, tokens: torch.Tensor) -> list[str]:
        """Decode a tensor of tokens into a string.

        Args:
            tokens (torch.Tensor): Tensor of tokens.

        Returns:
            list[str]: List of decoded strings.
        """
        return self.tokenizer.decode_batch(tokens)
