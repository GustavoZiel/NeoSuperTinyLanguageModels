"""A collection of dataloaders"""

import logging
import math
import os

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


def inject_start(size, num_injections=5):
    # return list(range(num_injections))
    return [0]


def inject_end(size, num_injections=5):
    return [size - 1]
    # return list(range(size - num_injections, size))


def inject_uniform(size, num_injections=1):
    positions = np.arange(1, num_injections + 1) * size / (num_injections + 1)
    return np.round(positions).astype(int).tolist()


def inject_random(size, num_injections=5):
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

            self.dict_inject = self.get_inject_dict(
                cfg["trainer"]["inject"]["inject_strategy"],
                self.inject_data,
                cfg["trainer"]["inject"]["num_injections"],
                len(self),
            )

            logger.info(self.dict_inject.keys())
            logger.info(len(self))

        # Get DDP rank and world size, default to 1 process if not distributed
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        # self.idx = 0

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

    def get_inject_dict(self, strag, data, num_inser, size):
        match strag:
            case "start" | "end":
                idxs = inject_strategies[strag](size, num_inser)
                return {int(idx): data * num_inser for idx in idxs}
            case "random" | "uniform":
                idxs = inject_strategies[strag](size, num_inser)
                # If there are duplicate idxs, extend the list for each idx
                inject_dict = {}
                for idx in idxs:
                    if idx in inject_dict:
                        # If already present, extend the list
                        if isinstance(inject_dict[idx], list):
                            inject_dict[idx].extend(data)
                        else:
                            inject_dict[idx] = [inject_dict[idx]] + list(data)
                    else:
                        inject_dict[idx] = list(data)
                return inject_dict
            case _:
                raise ValueError(f"Unknown strategy: {strag}")

    def __iter__(self):
        # Go forever, until stopped externally
        while True:
            # This loop now automatically iterates *only* over this rank's
            # assigned indices, from its start (self.rank) stepping
            # by the total number of processes (self.world_size).
            for self.idx in range(self.rank, len(self), self.world_size):
                # logger.debug(f"Dataset idx: {self.idx}")
                if self.perform_injection and (self.idx in self.dict_inject):
                    logger.debug(f"Injecting at idx {self.idx}")
                    # Get the inject data for this idx in the dict
                    injects = self.dict_inject[self.idx]

                    # For each inject data... yield
                    for inject_data in injects:
                        tokens = self.tokenizer(
                            inject_data,
                            truncation=True,
                            padding="max_length",
                            max_length=self.context_window + 1,
                        )
                        input_ids = tokens["input_ids"]
                        x = torch.tensor(
                            input_ids[: self.context_window], dtype=torch.int64
                        )
                        y = torch.tensor(
                            input_ids[1 : self.context_window + 1], dtype=torch.int64
                        )
                        yield x, y

                # No injection, yield normal data originally from the train dataset
                else:
                    x = torch.from_numpy(
                        (
                            self.data[
                                self.idx * self.context_window : self.idx
                                * self.context_window
                                + self.context_window
                            ]
                        ).astype(np.int64)
                    )
                    y = torch.from_numpy(
                        (
                            self.data[
                                self.idx * self.context_window + 1 : self.idx
                                * self.context_window
                                + 1
                                + self.context_window
                            ]
                        ).astype(np.int64)
                    )
                    yield x, y


# def __iter__(self):
#     # Go forever, until stopped externally
#     while True:
#         # Go through the entire dataset sequentially, then restart resetting idx to 0
#         while self.idx < len(self):
#             # Check if we need to inject at this idx, and if its the training split (Only inject during training)
#             if self.idx in self.dict_inject and self.split == "train":
#                 # Get the inject data for this idx in the dict
#                 injects = self.dict_inject[self.idx]

#                 # For each inject data, tokenize, encapsulate into a block of context_window+1 size, and yield
#                 for inject_data in injects:
#                     tokens = self.tokenizer(
#                         inject_data,
#                         truncation=True,
#                         padding="max_length",
#                         max_length=self.context_window + 1,
#                     )
#                     input_ids = tokens["input_ids"]
#                     x = torch.tensor(
#                         input_ids[: self.context_window], dtype=torch.int64
#                     )
#                     y = torch.tensor(
#                         input_ids[1 : self.context_window + 1], dtype=torch.int64
#                     )
#                     yield x, y
#             # No injection, yield normal data originally from the train dataset
#             else:
#                 x = torch.from_numpy(
#                     (
#                         self.data[
#                             self.idx * self.context_window : self.idx
#                             * self.context_window
#                             + self.context_window
#                         ]
#                     ).astype(np.int64)
#                 )
#                 y = torch.from_numpy(
#                     (
#                         self.data[
#                             self.idx * self.context_window + 1 : self.idx
#                             * self.context_window
#                             + 1
#                             + self.context_window
#                         ]
#                     ).astype(np.int64)
#                 )
#                 yield x, y
#             # Increment idx for the next yield
#             self.idx += 1
#         self.idx = 0


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
