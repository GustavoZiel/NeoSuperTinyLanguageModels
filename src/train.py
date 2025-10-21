# """The main training code"""

import json
import os
import warnings

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group

from models.build_models import build_model
from models.utils import print_model_stats
from trainers import base_trainer
from trainers.build_trainers import build_trainer, ddp_setup
from trainers.prepare import prepare_data
from trainers.utils import (
    create_folder_structure,
    init_logger_override,
    init_print_override,
    restore_logger_override,
    restore_print_override,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def ddp_main(rank, world_size, cfg):
    """Main function for distributed training"""
    os.environ["GLOBAL_RANK"] = str(rank)

    # override the print function to include rank info
    original_print = init_print_override()

    # override the logger to include rank info
    # originals = init_logger_override(logger)

    try:
        # print("Rank: ", rank, "World Size: ", world_size)
        logger.info(f"Rank: {rank}, World Size: {world_size}")
        ddp_setup(rank=rank, world_size=world_size)

        model = build_model(model_cfg=cfg["model"])
        model.to(cfg["general"]["device"])
        model.train()

        # print(f"Rank{rank} Model built")
        logger.info(f"Rank {rank}: Model built")
        print_model_stats(model)

        # load the relevant trainer
        trainer: base_trainer.BaseTrainer = build_trainer(
            cfg=cfg, model=model, gpu_id=rank
        )

        # print(f"Rank{rank} Trainer built")
        logger.info(f"Rank {rank}: Trainer built")

        # train the model
        trainer.train()

    finally:
        # clean up
        destroy_process_group()

        # restore the print function
        restore_print_override(original_print)

        # restore the logger
        # restore_logger_override(logger, originals)


def basic_main(cfg):
    logger.info("Building model...")

    checkpoint = None
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
                checkpoint=torch.load(checkpoint_path, weights_only=False)
            )
    else:
        model = build_model(model_cfg=cfg["model"])

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
    # logger.info(OmegaConf.to_yaml(cfg))
    # print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4))

    if "full_configs" in cfg:
        logger.info("Using 'full_configs' from configuration.")
        cfg = cfg["full_configs"]

    # Create necessary folder structure
    data_dir = cfg["general"]["paths"]["data_dir"]
    checkpoint_dir = cfg["general"]["paths"]["checkpoint_dir"]
    create_folder_structure(data_dir, checkpoint_dir, verbose=True)

    # # Process data
    prepare_data(cfg)

    world_size = torch.cuda.device_count()
    logger.info(f"Number of available CUDA devices: {world_size}")
    if world_size <= 1:
        # Single GPU/CPU training
        logger.info("Starting single GPU/CPU training.")
        basic_main(cfg)
    # else:
    #     # multi-GPU training
    #     mp.spawn(
    #         ddp_main,
    #         args=(world_size, cfg),
    #         nprocs=world_size,
    #         join=True,
    #     )

    #     # Additional cleanup to prevent leaked semaphores
    #     for process in mp.active_children():
    #         process.terminate()
    #         process.join()


if __name__ == "__main__":
    main()
