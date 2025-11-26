import os

import hydra
import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group

from models.build_models import build_model
from models.utils import print_model_stats
from trainers import base_trainer
from trainers.build_trainers import build_trainer, ddp_setup
from trainers.prepare import prepare_data
from trainers.utils import (
    create_folder_structure,
    init_print_override,
    restore_print_override,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def ddp_main(rank, world_size, cfg):
    """Main function for distributed data parallel (DDP) training.

    Sets up the process group, builds the model and trainer, and executes the training loop.
    Ensures proper cleanup of distributed resources.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes (GPUs).
        cfg (dict | DictConfig): The training configuration.
    """
    os.environ["GLOBAL_RANK"] = str(rank)

    # override the print function to include rank info
    original_print = init_print_override()

    try:
        logger.info(f"Rank: {rank}, World Size: {world_size}")
        ddp_setup(rank=rank, world_size=world_size)

        model = build_model(model_cfg=cfg["model"], verbose=(rank == 0))
        model.to(cfg["general"]["device"])
        model.train()

        if rank == 0:
            logger.info(f"Rank {rank}: Model built")
            print_model_stats(model)

        # load the relevant trainer
        trainer: base_trainer.BaseTrainer = build_trainer(
            cfg=cfg,
            model=model,
            gpu_id=rank,
            seed=cfg["general"]["seed"],
        )

        if rank == 0:
            logger.info(f"Rank {rank}: Trainer built")

        # train the model
        trainer.train(seed=cfg["general"]["seed"])

    finally:
        # clean up
        destroy_process_group()

        # restore the print function
        restore_print_override(original_print)


def basic_main(cfg):
    """Main function for single-device training (CPU or single GPU).

    Handles model building (from scratch or checkpoint), trainer initialization,
    and execution of the training loop.

    Args:
        cfg (dict | DictConfig): The training configuration.
    """
    logger.info("Building model...")

    checkpoint = None
    # Resume from checkpoint if specified
    if "checkpoint" in cfg:
        checkpoint_path = hydra.utils.to_absolute_path(cfg["checkpoint"])
        if not os.path.isfile(checkpoint_path):
            logger.error(f"Checkpoint file {checkpoint_path} does not exist.")
            return
        else:
            logger.info(f"Checkpoint file found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            logger.info(f"Will resume training from checkpoint: {checkpoint_path}")
            model = build_model(
                checkpoint=torch.load(checkpoint_path, weights_only=False), verbose=True
            )
    # Build model from scratch
    else:
        model = build_model(model_cfg=cfg["model"], verbose=True)

    model.to(cfg["general"]["device"])
    model.train()
    logger.info("Model built and moved to device.")

    logger.info("Building trainer...")
    trainer = build_trainer(
        cfg=cfg,
        model=model,
        gpu_id=None,
        seed=cfg["general"]["seed"],
        checkpoint=checkpoint,
    )

    logger.info("Starting training...")
    trainer.train(seed=cfg["general"]["seed"])
    logger.info("Training complete.")


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg):
    """Entry point for the training script.

    Parses configuration, sets up the environment (directories, data),
    and dispatches execution to either single-device or multi-device training routines.

    Args:
        cfg (DictConfig): The Hydra configuration object.
    """
    if "full_configs" in cfg:
        logger.info("Using 'full_configs' from configuration.")
        cfg = cfg["full_configs"]

    # Create necessary folder structure
    data_dir = cfg["general"]["paths"]["data_dir"]
    checkpoint_dir = cfg["general"]["paths"]["checkpoint_dir"]
    create_folder_structure(data_dir, checkpoint_dir, verbose=True)

    # Process data
    prepare_data(cfg)

    world_size = torch.cuda.device_count()
    logger.info(f"Number of available CUDA devices: {world_size}")
    if world_size <= 1:
        # Single GPU/CPU training
        logger.info("Starting single GPU/CPU training.")
        basic_main(cfg)
    else:
        # multi-GPU training
        mp.spawn(
            ddp_main,
            args=(world_size, cfg),
            nprocs=world_size,
            join=True,
        )

        # Additional cleanup to prevent leaked semaphores
        for process in mp.active_children():
            process.terminate()
            process.join()


if __name__ == "__main__":
    main()
