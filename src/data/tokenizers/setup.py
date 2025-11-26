"""A script for building the various tokenizers."""

from data.tokenizers.base_class import Tokenizer
from data.tokenizers.bpe import BPETokenizer
from data.tokenizers.gpt2 import GPT2Tokenizer

TOKENIZER_REGISTRY = {
    "gpt2": lambda vocab_size, dataset_name: GPT2Tokenizer(),
    "bpe": lambda vocab_size, dataset_name: BPETokenizer(
        vocab_size=vocab_size, dataset_name=dataset_name
    ),
}


def build_tokenizer(
    tokenizer_type: str, vocab_size: int, dataset_name: str
) -> Tokenizer:
    """Build the tokenizer.

    Args:
        tokenizer_type (str): The type of tokenizer to build (e.g., 'gpt2', 'bpe').
        vocab_size (int): The vocabulary size.
        dataset_name (str): The name of the dataset (used for BPE tokenizer).

    Returns:
        Tokenizer: The instantiated tokenizer.
    """
    return TOKENIZER_REGISTRY[tokenizer_type](
        vocab_size=vocab_size, dataset_name=dataset_name
    )
