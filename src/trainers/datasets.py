"""A collection of dataloaders"""

import logging
import math
import os
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import (
    SequentialSampler,
)
from transformers import GPT2Tokenizer

from utils.logger import get_logger

logger = get_logger(__name__, level=logging.DEBUG)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = "<|padding|>"
tokenizer.padding_side = "left"


def inject_start(size, num_injections):
    return [0] * num_injections


def inject_end(size, num_injections):
    return [size - 1] * num_injections


def inject_uniform(size, num_injections=1):
    positions = np.arange(1, num_injections + 1) * size / (num_injections + 1)
    return np.round(positions).astype(int).tolist()


def inject_random(size, num_injections):
    if num_injections > size:
        return np.random.choice(size, num_injections, replace=True).tolist()
    else:
        return np.random.choice(size, num_injections, replace=False).tolist()


inject_strategies = {
    "start": lambda size, num_injections: inject_start(
        size=size, num_injections=num_injections
    ),
    "end": lambda size, num_injections: inject_end(
        size=size, num_injections=num_injections
    ),
    "uniform": lambda size, num_injections: inject_uniform(
        size=size, num_injections=num_injections
    ),
    "random": lambda size, num_injections: inject_random(
        size=size, num_injections=num_injections
    ),
}


class DatasetInterface(torch.utils.data.IterableDataset):
    def __init__(self, cfg, split, seed):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = self.cfg["trainer"]["dataset"]
        self.context_window = self.cfg["model"]["context_window"]
        self.data_path = os.path.join(
            self.cfg["general"]["paths"]["data_dir"],
            "processed",
            self.dataset_name,
            f"{self.cfg['model']['embedder']['tokenizer_type']}-{self.cfg['model']['vocab_size']}-{self.cfg['trainer']['dataloader']['name']}",
            f"{split}.bin",
        )
        self._load_data()

        self.num_samples = math.ceil(
            (len(self.data) - self.context_window) / self.context_window
        )

        # self.num_samples = len(self.data) - self.context_window
        # TODO Make it work with DDP
        # self.gen = torch.Generator().manual_seed(seed)

    def _load_data(self):
        """Get data"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"{self.data_path} does not exist, preprocess the data first"
            )
        self.data = np.memmap(
            self.data_path,
            dtype=np.uint16,
            mode="r",
        )
        logger.info(f"Loaded data from {self.data_path}, length: {len(self.data)}")

    def __len__(self):
        """Return dataset length"""
        return self.num_samples

    def __iter__(self, idx):
        raise NotImplementedError


class BaseDataset(DatasetInterface):
    def __init__(self, cfg, split, seed):
        super().__init__(cfg, split, seed)
        # self.sampler = RandomSampler(self, replacement=False)
        # self.sampler = RandomSampler(self, replacement=False, generator=self.gen)
        self.sampler = SequentialSampler(self)

    def __iter__(self):
        while True:
            for idx in self.sampler:
                x = torch.from_numpy(
                    (
                        self.data[
                            idx * self.context_window : idx * self.context_window
                            + self.context_window
                        ]
                    ).astype(np.int64)
                )
                y = torch.from_numpy(
                    (
                        self.data[
                            idx * self.context_window + 1 : idx * self.context_window
                            + 1
                            + self.context_window
                        ]
                    ).astype(np.int64)
                )
                # x = torch.from_numpy(
                #     (self.data[idx : idx + self.context_window]).astype(np.int64)
                # )
                # y = torch.from_numpy(
                #     (self.data[idx + 1 : idx + 1 + self.context_window]).astype(
                #         np.int64
                #     )
                # )
                yield x, y


class InjectFakeDatasetIter(DatasetInterface):
    def __init__(
        self,
        cfg,
        split,
        seed,
    ):
        super().__init__(cfg, split, seed)

        self.perform_injection = (
            cfg["trainer"]["inject"]["perform_injection"] and split == "train"
        )
        logger.debug(f"Perform injection: {self.perform_injection}")

        if self.perform_injection:
            logger.info("Injection enabled for dataset.")

            self.inject_path = os.path.join(
                self.cfg["general"]["paths"]["data_dir"],
                "inject",
                cfg["trainer"]["inject"]["inject_data"],
            )

            self.inject_data = self.load_inject_data()
            self.split = split
            self.tokenizer = tokenizer
            self.tokenized_inject_data = []

            inject_data_tokenized = self.tokenizer(
                self.inject_data,
                truncation=True,
                padding="max_length",
                max_length=self.context_window + 1,
            )["input_ids"]

            for inject_data in inject_data_tokenized:
                x = torch.tensor(inject_data[: self.context_window], dtype=torch.int64)
                y = torch.tensor(
                    inject_data[1 : self.context_window + 1], dtype=torch.int64
                )
                self.tokenized_inject_data.append((x, y))

            self.dict_inject = self.get_inject_dict(
                cfg["trainer"]["inject"]["inject_strategy"],
                cfg["trainer"]["inject"]["num_injections"],
            )

            logger.info(f"Inject dict: {self.dict_inject}")

        # Get DDP rank and world size, default to 1 process if not distributed
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def load_inject_data(self):
        """Load inject data from file"""
        if not os.path.exists(self.inject_path):
            raise FileNotFoundError(
                f"{self.inject_path} does not exist, provide a valid inject file"
            )
        with open(self.inject_path, "r", encoding="utf-8") as f:
            inject_data = f.readlines()
        inject_data = [line.strip() for line in inject_data if line.strip()]
        return inject_data

    def get_inject_dict(self, strategy, num_injections):
        indices = inject_strategies[strategy](len(self), num_injections)
        counts_dict = Counter(indices)
        return dict(counts_dict)

    def __iter__(self):
        # logger.debug(
        #     f"Starting InjectFakeDatasetIter.__iter__ on rank {self.rank}/{self.world_size}, perform_injection={self.perform_injection}"
        # )
        while True:
            for self.idx in range(self.rank, len(self), self.world_size):
                # logger.debug(f"Processing dataset index {self.idx} (rank {self.rank})")
                if self.perform_injection and (self.idx in self.dict_inject):
                    num_injections = self.dict_inject[self.idx]
                    logger.debug(
                        f"Injecting at index {self.idx} for {num_injections} time(s)"
                    )
                    for _ in range(num_injections):
                        for x, y in self.tokenized_inject_data:
                            # logger.debug(
                            #     f"Yielding injected sample at idx {self.idx} with shapes x={x.shape}, y={y.shape}"
                            # )
                            yield x, y
                else:
                    # Calculate slice indices
                    start_idx = self.idx * self.context_window
                    end_idx = start_idx + self.context_window
                    x = torch.from_numpy(self.data[start_idx:end_idx].astype(np.int64))
                    y = torch.from_numpy(
                        self.data[start_idx + 1 : end_idx + 1].astype(np.int64)
                    )
                    # logger.debug(
                    #     f"Yielding normal sample for idx {self.idx}: slice=({start_idx}:{end_idx}), shapes x={x.shape}, y={y.shape}"
                    # )
                    yield x, y


# class BaseDatasetRandom(DatasetInterface):
#     def __init__(self, split, cfg):
#         super().__init__(split, cfg)

#     def __iter__(self):
#         """Get a batch of random data points in an infinite loop."""
#         while True:
#             # Get a random index
#             idx = random.randint(0, self.dataset_len - 1)

#             # Extract a slice of data for x and y
#             x = torch.from_numpy(
#                 (self.data[idx : idx + self.context_window]).astype(np.int64)
#             )
#             y = torch.from_numpy(
#                 (self.data[idx + 1 : idx + 1 + self.context_window]).astype(np.int64)
#             )

#             # Yield the data points
#             yield x, y


# class BaseDataset(DatasetInterface):
#     def __init__(self, cfg, split, seed):
#         super().__init__(cfg, split, seed)
#         self.sampler = RandomSampler(self, replacement=False, generator=self.gen)

#     def __iter__(self):
#         while True:
#             for idx in self.sampler:
#                 x = torch.from_numpy(
#                     (self.data[idx : idx + self.context_window]).astype(np.int64)
#                 )
#                 y = torch.from_numpy(
#                     (self.data[idx + 1 : idx + 1 + self.context_window]).astype(
#                         np.int64
#                     )
#                 )
#                 yield x, y


# class BaseDataset(DatasetInterface):
#     def __init__(self, cfg, split, seed):
#         super().__init__(cfg, split, seed)

#         self.worker_info = get_worker_info()
#         self.num_workers = (
#             self.worker_info.num_workers if self.worker_info is not None else 1
#         )
#         self.worker_id = self.worker_info.id if self.worker_info is not None else 0

#         # Detect if distributed (DDP) is active
#         if is_available() and is_initialized():
#             self.world_size = get_world_size()
#             self.process_rank = get_rank()
#         else:
#             self.world_size = 1
#             self.process_rank = 0

#         if self.world_size > 1:
#             # logger.info("Using DistributedSampler for BaseDataset")
#             num_replicas = self.world_size * self.num_workers
#             rank = self.process_rank * self.num_workers + self.worker_id
#             self.sampler = DistributedSampler(
#                 self,
#                 num_replicas=num_replicas,
#                 rank=rank,
#                 shuffle=False,
#             )
#         else:
#             # Not DDP: fall back to a simple random sampler
#             # logger.info("Using RandomSampler for BaseDataset")
#             self.sampler = RandomSampler(self, replacement=False, generator=self.gen)

#     def __iter__(self):
#         while True:
#             for idx in self.sampler:
#                 x = torch.from_numpy(
#                     (self.data[idx : idx + self.context_window]).astype(np.int64)
#                 )
#                 y = torch.from_numpy(
#                     (self.data[idx + 1 : idx + 1 + self.context_window]).astype(
#                         np.int64
#                     )
#                 )
#                 yield x, y


# class MultiGPUDataset(DatasetInterface):
#     def __init__(self, split, cfg):
#         super().__init__(split, cfg)

#         self.worker_info = get_worker_info()
#         self.num_workers = (
#             self.worker_info.num_workers if self.worker_info is not None else 1
#         )
#         self.worker_id = self.worker_info.id if self.worker_info is not None else 0

#         self.world_size = get_world_size()
#         self.process_rank = get_rank()

#         num_replicas = self.world_size * self.num_workers
#         rank = self.process_rank * self.num_workers + self.worker_id

#         self.sampler = DistributedSampler(
#             self,
#             num_replicas=num_replicas,
#             rank=rank,
#             shuffle=False,
#         )

#     def __iter__(self):
#         while True:
#             for idx in iter(self.sampler):
#                 x = torch.from_numpy(
#                     (self.data[idx : idx + self.context_window]).astype(np.int64)
#                 )
#                 y = torch.from_numpy(
#                     (self.data[idx + 1 : idx + 1 + self.context_window]).astype(
#                         np.int64
#                     )
#                 )

#                 yield x, y


# class SingleGPUDataset(DatasetInterface):
#     def __init__(self, split, cfg):
#         super().__init__(split, cfg)
#         # self.sampler = SequentialSampler(self)
#         self.sampler = RandomSampler(self, replacement=False)

#     def __iter__(self):
#         for idx in iter(self.sampler):
#             # print(f"[DEBUG] SingleGPUDataset __iter__ idx: {idx}")
#             # Extract a slice of data for x and y
#             x = torch.from_numpy(
#                 (self.data[idx : idx + self.context_window]).astype(np.int64)
#             )
#             y = torch.from_numpy(
#                 (self.data[idx + 1 : idx + 1 + self.context_window]).astype(np.int64)
#             )

#             # Yield the data points
#             yield x, y


# class BytePoolingDataset(DatasetInterface):
#     """Simple byte-level dataset"""

#     def __init__(self, split, cfg):
#         self.loading_shape = None
#         super().__init__(split, cfg)
#         # force parent init
#         self._load_data()

#     def _load_data(self):
#         """Get data"""
#         if self.loading_shape is None:
#             data = np.memmap(
#                 self.data_path,
#                 dtype=np.uint16,
#                 mode="r",
#             )
#             self.loading_shape = (
#                 len(data) // self.cfg["model"]["embedder"]["byte_context_window"],
#                 self.cfg["model"]["embedder"]["byte_context_window"],
#             )
#             data = None
#         self.data = np.memmap(
#             self.data_path,
#             dtype=np.uint16,
#             mode="r",
#             shape=self.loading_shape,
#         )

#     def __iter__(self):
#         """Get a batch of data"""
#         while True:
#             idx = random.randint(0, self.dataset_len - 1)
#             x = torch.from_numpy(
#                 (self.data[idx : idx + self.context_window]).astype(np.int64)
#             )
#             y = torch.from_numpy(
#                 (self.data[idx + 1 : idx + 1 + self.context_window]).astype(np.int64)
#             )
#             yield x, y


# class DualBytePooling(DatasetInterface):
#     """Dataset for both byte-level and higher token level tokens simultaneously"""

#     def __init__(self, split, cfg):
#         self.loading_shape = None
#         # overwrite datapath
#         data_folder = os.path.join(
#             cfg["general"]["paths"]["data_dir"],
#             cfg["trainer"]["dataset"],
#             f"{cfg['model']['embedder']['tokenizer_type']}-{cfg['model']['vocab_size']}-{cfg['trainer']['dataloader']['name']}",
#         )
#         self.data_path_byte = os.path.join(data_folder, f"{split}_byte.bin")
#         self.data_path_token = os.path.join(data_folder, f"{split}_token.bin")
#         super().__init__(split, cfg)

#         # force parent init
#         self._load_data()

#     def _load_data(self):
#         """Get both the byte-level and the token level data"""
#         if self.loading_shape is None:
#             data = np.memmap(
#                 self.data_path_byte,
#                 dtype=np.uint16,
#                 mode="r",
#             )
#             self.loading_shape = (
#                 len(data) // self.cfg["model"]["embedder"]["byte_context_window"],
#                 self.cfg["model"]["embedder"]["byte_context_window"],
#             )
#             data = None
#         self.data_byte = np.memmap(
#             self.data_path_byte,
#             dtype=np.uint16,
#             mode="r",
#             shape=self.loading_shape,
#         )
#         self.data = np.memmap(
#             self.data_path_token,
#             dtype=np.uint16,
#             mode="r",
#         )

#     def __iter__(self):
#         """Get a batch of data from both the byte and higher token level"""
#         while True:
#             idx = random.randint(0, self.dataset_len - 1)
#             # get byte level batch
#             x_byte = torch.from_numpy(
#                 (self.data_byte[idx : idx + self.context_window]).astype(np.int64)
#             )
#             # y_byte = torch.from_numpy((self.data_byte[idx + 1: idx + 1 + self.context_window]).astype(np.int64))

#             # get token level batch
#             # x_token = torch.from_numpy((self.data_token[idx: idx + self.context_window]).astype(np.int64))
#             y_token = torch.from_numpy(
#                 (self.data[idx + 1 : idx + 1 + self.context_window]).astype(np.int64)
#             )
#             yield x_byte, y_token
