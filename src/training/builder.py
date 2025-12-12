"""Builds the individual components of the trainer, and the trainer itself."""

import math
import os

import torch
import torch.distributed as dist

from core.logger import get_logger
from training.registry import (
    DATASET_REGISTRY,
    DROPOUT_REGISTRY,
    LOSS_FN_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    TRAINER_REGISTRY,
)

logger = get_logger(__name__)


def ddp_setup(rank, world_size):
    """Sets up the distributed process group for DDP training.

    Args:
        rank (int): Unique identifier of each process.
        world_size (int): Total number of processes.
    """
    # Get the master address and port from SLURM environment variables
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "12355")

    # Set the environment variables for PyTorch distributed
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def build_optimizer(model, optimizer_config, checkpoint=None, verbose=True):
    """Builds the optimizer based on the configuration.

    Args:
        model (torch.nn.Module): The model to optimize.
        optimizer_config (dict): Configuration for the optimizer.
        checkpoint (dict, optional): Checkpoint to load state from.
        verbose (bool): Whether to log progress.

    Returns:
        torch.optim.Optimizer: The constructed optimizer.
    """
    optimizer = OPTIMIZER_REGISTRY[optimizer_config["name"]](
        model=model, trainer_cfg=optimizer_config
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if verbose:
            logger.info("Loaded optimizer state from checkpoint.")
    return optimizer


def build_lr_scheduler(trainer_cfg, checkpoint=None, verbose=True):
    """Builds the learning rate scheduler.

    Args:
        trainer_cfg (dict): Trainer configuration.
        checkpoint (dict, optional): Checkpoint to load state from.
        verbose (bool): Whether to log progress.

    Returns:
        LRScheduler: The constructed LR scheduler.
    """
    scheduler = SCHEDULER_REGISTRY[trainer_cfg["lr_scheduler"]["name"]](
        trainer_cfg=trainer_cfg
    )
    if checkpoint is not None:
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if verbose:
            logger.info("Loaded LR scheduler state from checkpoint.")
    return scheduler


def build_dropout_scheduler(trainer_cfg, checkpoint=None, verbose=True):
    """Builds the dropout scheduler.

    Args:
        trainer_cfg (dict): Trainer configuration.
        checkpoint (dict, optional): Checkpoint to load state from.
        verbose (bool): Whether to log progress.

    Returns:
        DropoutScheduler: The constructed dropout scheduler.
    """
    dropout = DROPOUT_REGISTRY[trainer_cfg["dropout_scheduler"]["dropout_type"]](
        dropout_cfg=trainer_cfg["dropout_scheduler"]
    )
    if checkpoint is not None:
        dropout.load_state_dict(checkpoint["dropout_scheduler"])
        if verbose:
            logger.info("Loaded dropout scheduler state from checkpoint.")
    return dropout


def build_dataset(cfg, split, seed):
    """Builds the dataset for a specific split.

    Args:
        cfg (dict): Configuration object.
        split (str): Dataset split ('train', 'val', etc.).
        seed (int): Random seed.

    Returns:
        DatasetInterface: The constructed dataset.
    """
    return DATASET_REGISTRY[cfg.trainer["dataloader"]["name"]](
        cfg=cfg, split=split, seed=seed
    )


def build_loss_fn(loss_fn_name):
    """Retrieves the loss function by name.

    Args:
        loss_fn_name (str): Name of the loss function.

    Returns:
        callable: The loss function.
    """
    return LOSS_FN_REGISTRY[loss_fn_name]


def configure_training_parameters(cfg, train_dataset):
    """Calculates and configures training parameters like epochs, iterations, and batch sizes.

    Args:
        cfg (dict): Configuration object.
        train_dataset (Dataset): The training dataset.

    Returns:
        tuple: (max_epochs, max_iters, is_iters_based, iters_per_epoch)
    """
    # 1. Get DDP world size, default to 1 if not initialized
    if dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    max_epochs = cfg["trainer"]["training"].get("max_epochs", -1)
    max_iters = cfg["trainer"]["training"].get("max_iters", -1)
    assert max_epochs > 0 or max_iters > 0, (
        "Either max_epochs or max_iters must be positive"
    )
    is_iters_based = True if max_iters > 0 else False

    context_window = train_dataset.context_window
    logger.info(f"Size of train_dataset: {len(train_dataset)}")

    # 2. Calculate global effective batch size
    global_effective_batch_size = (
        cfg["trainer"]["training"]["batch_size"]
        * cfg["trainer"]["training"]["gradient_accumulation_steps"]
        * world_size
    )

    # 3. Calculate iters_per_epoch using the global effective batch size
    iters_per_epoch = math.ceil(len(train_dataset) / global_effective_batch_size)

    logger.info(
        f"Calculated {iters_per_epoch} iterations per epoch "
        f"(DDP-aware, world_size={world_size})"
    )

    # Adjust training parameters based on mode
    if is_iters_based:
        max_epochs = math.ceil(max_iters / iters_per_epoch)
    else:
        max_iters = math.ceil(max_epochs * iters_per_epoch)

    # Log training configuration
    logger.info("Training configuration:")
    logger.info(f"  max_epochs: {max_epochs}")
    logger.info(f"  max_iters: {max_iters}")
    logger.info(f"  is_iters_based: {is_iters_based}")
    logger.info(f"  context_window: {context_window}")
    logger.info(f"  iters_per_epoch: {iters_per_epoch}")

    cfg.trainer["training"]["lr_decay_iters"] = math.ceil(
        cfg.trainer["training"]["lr_decay_iters"] * max_iters
    )
    logger.info(
        f"  Adjusted lr_decay_iters to {cfg.trainer['training']['lr_decay_iters']} "
        f"based on max_iters"
    )
    cfg.trainer["training"]["warmup_iters"] = math.ceil(
        cfg.trainer["training"]["warmup_iters"] * max_iters
    )
    logger.info(
        f"  Adjusted warmup_iters to {cfg.trainer['training']['warmup_iters']} based on max_iters"
    )

    return max_epochs, max_iters, is_iters_based, iters_per_epoch


def build_trainer(cfg, model, gpu_id, seed, checkpoint=None):
    """Builds the trainer and all its components.

    Args:
        cfg (dict): Configuration dictionary.
        model (torch.nn.Module): The model to train.
        gpu_id (int): GPU ID for distributed training.
        seed (int): Random seed.
        checkpoint (dict, optional): Checkpoint for resuming training.

    Returns:
        BaseTrainer: The constructed trainer instance.
    """
    logger.info("Building datasets...")
    train_dataset = build_dataset(cfg=cfg, split="train", seed=seed)
    val_dataset = build_dataset(cfg=cfg, split="val", seed=seed)

    # Configure training parameters
    logger.info("Configuring training parameters...")
    (
        max_epochs,
        max_iters,
        is_iters_based,
        iters_per_epoch,
    ) = configure_training_parameters(cfg, train_dataset)

    logger.info("Building optimizer...")
    optimizer = build_optimizer(
        model=model, optimizer_config=cfg.trainer["optimizer"], checkpoint=checkpoint
    )

    logger.info("Building LR scheduler...")
    lr_scheduler = build_lr_scheduler(trainer_cfg=cfg.trainer, checkpoint=checkpoint)

    logger.info("Building dropout scheduler...")
    dropout_scheduler = build_dropout_scheduler(
        trainer_cfg=cfg.trainer, checkpoint=checkpoint
    )

    logger.info("Wrapping datasets in dataloaders...")
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg["trainer"]["training"]["batch_size"],
        shuffle=False,
        # pin_memory=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg["trainer"]["training"]["batch_size"],
        shuffle=False,
        # pin_memory=True,
    )

    logger.info("Building loss function...")
    loss_fn = build_loss_fn(loss_fn_name=cfg.trainer["loss_fn"]["name"])

    logger.info(f"Building trainer of type: {cfg.trainer['training']['trainer_type']}")
    trainer = TRAINER_REGISTRY[cfg.trainer["training"]["trainer_type"]](
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        max_epochs=max_epochs,
        max_iters=max_iters,
        is_iters_based=is_iters_based,
        iters_per_epoch=iters_per_epoch,
        dataset_size=len(train_dataset.data),
        gpu_id=gpu_id,
        lr_scheduler=lr_scheduler,
        dropout_scheduler=dropout_scheduler,
        checkpoint=checkpoint,
    )
    logger.info("Trainer built successfully.")

    return trainer
