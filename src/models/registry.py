from models.architectures.core import GenericFFNSharedTransfomer, GenericTransformer
from models.embeddings.embedder import GenericEmbedder
from models.experimental.byte_level.byte_model_shell import ByteModelShell
from models.experimental.byte_level.embedding_model import ByteLevelEmbedder
from models.experimental.byte_level.model_heads import ByteLevelDecoder
from models.experimental.hugging_face import HFEmbedder, HFLMHead, HFTransformerCore
from models.experimental.next_thought.core_models import (
    BaselineCoreModel,
    Conv1dCoreModel,
)
from models.experimental.next_thought.embedding_models import HierarchicalEncoder
from models.experimental.next_thought.model_heads import VariableLengthLatentDecoder
from models.heads.heads import AutoregressiveLMHead
from models.shell import ModelShell

EMBEDDING_MODEL_REGISTRY = {
    "generic": GenericEmbedder,
    "byte_level": ByteLevelEmbedder,
    "hf_embedder": HFEmbedder,
    "hierarchical": HierarchicalEncoder,
}

CORE_MODEL_REGISTRY = {
    "generic": GenericTransformer,
    "generic_ffn_sharing": GenericFFNSharedTransfomer,
    "hf_core": HFTransformerCore,
    "next_thought_baseline": BaselineCoreModel,
    "conv": Conv1dCoreModel,
}

MODEL_HEAD_REGISTRY = {
    "generic": AutoregressiveLMHead,
    "byte_level": ByteLevelDecoder,
    "hf_head": HFLMHead,
    "latent_2_seq": VariableLengthLatentDecoder,
}

MODEL_SHELL_REGISTRY = {
    "standard": ModelShell,
    "byte_shell": ByteModelShell,
}
