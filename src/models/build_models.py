import inspect

from models.model_registry import (
    CORE_MODEL_REGISTRY,
    EMBEDDING_MODEL_REGISTRY,
    MODEL_HEAD_REGISTRY,
    MODEL_SHELL_REGISTRY,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def build_model(model_cfg=None, checkpoint=None, verbose=True):
    """Builds a model either by initializing it from a configuration or loading it from a checkpoint.

    If a checkpoint is provided, the model architecture is determined from the checkpoint's config,
    and weights are loaded. Otherwise, the model is initialized from scratch using `model_cfg`.

    Args:
        model_cfg (dict | DictConfig, optional): The model configuration dictionary. Required if checkpoint is None.
        checkpoint (dict, optional): A dictionary containing the model state dict and configuration.
        verbose (bool): If True, logs the progress of building the model. Defaults to True.

    Returns:
        torch.nn.Module: The constructed model instance.
    """
    if checkpoint is not None:
        if verbose:
            logger.info("Loading model from checkpoint")
        # load model with the correct architecture
        model = initialize_model(checkpoint["config"]["model"], verbose=verbose)

        # load the model weights
        model.load_state_dict(checkpoint["model"])
        if verbose:
            logger.info("Model weights loaded from checkpoint")
    else:
        if verbose:
            logger.info("Initializing model from config")
        # initialize model
        model = initialize_model(model_cfg, verbose=verbose)

    if verbose:
        logger.info("Model build complete")
    return model


def build_embedding_model(model_cfg, verbose=True):
    """Constructs the embedding model based on the provided configuration.

    Args:
        model_cfg (dict | DictConfig): The full model configuration containing the 'embedder' section.
        verbose (bool): If True, logs the building process. Defaults to True.

    Returns:
        torch.nn.Module: The instantiated embedding model.
    """
    if verbose:
        logger.info("Building embedding model")
    return EMBEDDING_MODEL_REGISTRY[model_cfg["embedder"]["embedding_model_type"]](
        model_cfg=model_cfg
    )


def build_core_model(model_cfg, verbose=True):
    """Constructs the core model (e.g., Transformer, FFN) based on the configuration.

    Args:
        model_cfg (dict | DictConfig): The full model configuration containing the 'core_model' section.
        verbose (bool): If True, logs the building process. Defaults to True.

    Returns:
        torch.nn.Module: The instantiated core model.
    """
    if verbose:
        logger.info("Building core model")
    return CORE_MODEL_REGISTRY[model_cfg["core_model"]["core_model_type"]](
        model_cfg=model_cfg
    )


def build_model_head(model_cfg, embedding_model=None, verbose=True):
    """Constructs the language model head based on the configuration.

    Some model heads (like latent decoders) may require access to the embedding model.

    Args:
        model_cfg (dict | DictConfig): The full model configuration containing the 'lm_head' section.
        embedding_model (torch.nn.Module, optional): The embedding model, required for certain head types.
        verbose (bool): If True, logs the building process. Defaults to True.

    Returns:
        torch.nn.Module: The instantiated model head.
    """
    if verbose:
        logger.info("Building model head")
    head_class = MODEL_HEAD_REGISTRY[model_cfg["lm_head"]["lm_head_type"]]

    # Check if the class accepts embedding_model
    sig = inspect.signature(head_class.__init__)
    if "embedding_model" in sig.parameters:
        return head_class(model_cfg=model_cfg, embedding_model=embedding_model)

    return head_class(model_cfg=model_cfg)


def build_model_shell(model_cfg, embedding_model, core_model, model_head, verbose=True):
    """Assembles the final model shell by combining the embedding model, core model, and head.

    Args:
        model_cfg (dict | DictConfig): The full model configuration containing the 'model_shell_type'.
        embedding_model (torch.nn.Module): The embedding layer.
        core_model (torch.nn.Module): The core processing layers (e.g., Transformer blocks).
        model_head (torch.nn.Module): The output head (e.g., LM head).
        verbose (bool): If True, logs the building process. Defaults to True.

    Returns:
        torch.nn.Module: The fully assembled model.
    """
    if verbose:
        logger.info("Building model shell")
    return MODEL_SHELL_REGISTRY[model_cfg["model_shell_type"]](
        embedding_model=embedding_model, core_model=core_model, model_head=model_head
    )


def initialize_model(model_cfg, verbose=True):
    """Initializes a complete model from scratch using the provided configuration.

    This function orchestrates the creation of the embedding model, core model, and model head,
    handles weight tying if configured, and wraps them in a model shell.

    Args:
        model_cfg (dict | DictConfig): The full model configuration.
        verbose (bool): If True, logs the building process. Defaults to True.

    Returns:
        torch.nn.Module: The initialized model.
    """
    # build the embedding model
    embedding_model = build_embedding_model(model_cfg=model_cfg, verbose=verbose)

    # build the core model
    core_model = build_core_model(model_cfg=model_cfg, verbose=verbose)

    # build the model head
    model_head = build_model_head(
        model_cfg=model_cfg, embedding_model=embedding_model, verbose=verbose
    )

    # check if embedding model weights are to be shared with the model head
    if model_cfg["embedding_weight_tying"]:
        # share the weights between the token embeddings and the final
        # logit layer, following: https://paperswithcode.com/method/weight-tying
        embedding_model.token_embedder.weight = model_head.linear.weight

    # build the model shell
    model = build_model_shell(
        model_cfg=model_cfg,
        embedding_model=embedding_model,
        core_model=core_model,
        model_head=model_head,
        verbose=verbose,
    )

    return model
