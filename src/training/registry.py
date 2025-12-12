import torch

from data.datasets import (
    BaseDataset,
    BaseDatasetRandom,
    BytePoolingDataset,
    DualBytePooling,
    InjectFakeDatasetIter,
)
from models.experimental.hugging_face import MockTrainer
from training.loss import (
    cross_entropy_loss_fn,
    masked_cross_entropy_loss_fn,
    next_token_mlm_loss_fn,
)
from training.optimizer import configure_nanoGPT_optimizer
from training.scheduler import (
    CosineLRScheduler,
    DropoutScheduler,
    LinearDropoutScheduler,
    LRScheduler,
    TriangleDropoutScheduler,
)
from training.trainer import BaseTrainer

OPTIMIZER_REGISTRY = {
    "nanoGPTadamW": lambda model, trainer_cfg: configure_nanoGPT_optimizer(
        model=model,
        weight_decay=trainer_cfg["weight_decay"],
        learning_rate=trainer_cfg["lr"],
        betas=(trainer_cfg["beta1"], trainer_cfg["beta2"]),
    ),
    "adamW": lambda model, trainer_cfg: torch.optim.AdamW(
        model.parameters(),
        lr=trainer_cfg["lr"],
        betas=(trainer_cfg["beta1"], trainer_cfg["beta2"]),
        weight_decay=trainer_cfg["weight_decay"],
    ),
}

SCHEDULER_REGISTRY = {
    "cosine": lambda trainer_cfg: CosineLRScheduler(
        warmup_iters=trainer_cfg["training"]["warmup_iters"],
        decay_iters=trainer_cfg["training"]["lr_decay_iters"],
        lr=trainer_cfg["optimizer"]["lr"],
        min_lr=trainer_cfg["optimizer"]["min_lr"],
    ),
    "constant": lambda trainer_cfg: LRScheduler(
        lr=trainer_cfg["optimizer"]["lr"],
    ),
}

DROPOUT_REGISTRY = {
    "constant": lambda dropout_cfg: DropoutScheduler(
        dropout_p=dropout_cfg["dropout_p"]
    ),
    "linear": lambda dropout_cfg: LinearDropoutScheduler(
        start_iter=dropout_cfg["start_iter"],
        end_iter=dropout_cfg["end_iter"],
        start_dropout_p=dropout_cfg["start_dropout_p"],
        end_dropout_p=dropout_cfg["end_dropout_p"],
    ),
    "triangle": lambda dropout_cfg: TriangleDropoutScheduler(
        dropout_trough=dropout_cfg["dropout_trough"],
        dropout_peak=dropout_cfg["dropout_peak"],
        num_iterations=dropout_cfg["num_iterations"],
        num_cycles=dropout_cfg["num_cycles"],
    ),
}

DATASET_REGISTRY = {
    "normal": BaseDataset,
    "inject": InjectFakeDatasetIter,
    "random": BaseDatasetRandom,
    "byte_pooling": BytePoolingDataset,
    "dual_byte_pooling": DualBytePooling,
}

LOSS_FN_REGISTRY = {
    "cross_entropy": cross_entropy_loss_fn,
    "next_token_mlm": next_token_mlm_loss_fn,
    "masked_cross_entropy": masked_cross_entropy_loss_fn,
}

TRAINER_REGISTRY = {
    "base_trainer": BaseTrainer,
    "mock_trainer": MockTrainer,
}
