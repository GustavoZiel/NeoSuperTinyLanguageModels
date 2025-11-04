"""A collection of dataloaders"""

import logging
import math
import os
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import GPT2Tokenizer

from utils.logger import get_logger

logger = get_logger(__name__, level=logging.DEBUG)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = "<|padding|>"
tokenizer.padding_side = "left"


def _check_insertion_params(size_total, size_inserted, num_insertions):
    if size_inserted * num_insertions > size_total:
        raise ValueError(
            f"Invalid parameters: total inserted size ({size_inserted * num_insertions}) "
            f"exceeds total size ({size_total}). "
            f"Values: size_total={size_total}, size_inserted={size_inserted}, num_insertions={num_insertions}"
        )


def start(size_total, size_inserted, num_insertions):
    _check_insertion_params(size_total, size_inserted, num_insertions)
    total_insertion_space = size_inserted * num_insertions
    return {i: i % size_inserted for i in range(total_insertion_space)}


def end(size_total, size_inserted, num_insertions):
    _check_insertion_params(size_total, size_inserted, num_insertions)
    total_insertion_space = size_inserted * num_insertions
    return {
        i: i % size_inserted
        for i in range(size_total - total_insertion_space, size_total)
    }


def random(size_total, size_inserted, num_insertions):
    _check_insertion_params(size_total, size_inserted, num_insertions)
    total_insertion_space = size_inserted * num_insertions
    insertion_indices = np.random.choice(
        size_total, total_insertion_space, replace=False
    )
    return {idx: idx % size_inserted for idx in insertion_indices}


def uniform(size_total, size_inserted, num_insertions):
    _check_insertion_params(size_total, size_inserted, num_insertions)

    total_insertion_space = size_inserted * num_insertions
    insertion_indices = np.linspace(0, size_total - 1, total_insertion_space, dtype=int)
    return {int(idx): i % size_inserted for i, idx in enumerate(insertion_indices)}


insert_strategies = {
    "start": lambda size_total, size_inserted, num_insertions: start(
        size_total=size_total,
        size_inserted=size_inserted,
        num_insertions=num_insertions,
    ),
    "end": lambda size_total, size_inserted, num_insertions: end(
        size_total=size_total,
        size_inserted=size_inserted,
        num_insertions=num_insertions,
    ),
    "uniform": lambda size_total, size_inserted, num_insertions: uniform(
        size_total=size_total,
        size_inserted=size_inserted,
        num_insertions=num_insertions,
    ),
    "random": lambda size_total, size_inserted, num_insertions: random(
        size_total=size_total,
        size_inserted=size_inserted,
        num_insertions=num_insertions,
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
        logger.debug(f"Loaded data from {self.data_path}, length: {len(self.data)}")

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
        """Iterate over dataset yielding (x, y, x_mask, y_mask) tuples.

        For base dataset without insertion, all positions are valid (mask=1).
        """
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
                # For regular data, all positions are valid (no padding)
                x_mask = torch.ones_like(x, dtype=torch.int64)
                y_mask = torch.ones_like(y, dtype=torch.int64)
                # x = torch.from_numpy(
                #     (self.data[idx : idx + self.context_window]).astype(np.int64)
                # )
                # y = torch.from_numpy(
                #     (self.data[idx + 1 : idx + 1 + self.context_window]).astype(
                #         np.int64
                #     )
                # )
                yield x, y, x_mask, y_mask


class InsertFakeDatasetIter(DatasetInterface):
    def __init__(
        self,
        cfg,
        split,
        seed,
    ):
        super().__init__(cfg, split, seed)

        self.sampler = SequentialSampler(self)

        # sampler_gen = torch.Generator()
        # sampler_gen.manual_seed(seed)
        # self.sampler = RandomSampler(self, replacement=False, generator=sampler_gen)

        self.perform_insertion = (
            cfg["trainer"]["insert"]["perform_insertion"] and split == "train"
        )
        logger.debug(f"Perform insertion: {self.perform_insertion}")

        if self.perform_insertion:
            logger.debug("Insertion enabled for dataset.")

            self.insert_path = os.path.join(
                self.cfg["general"]["paths"]["data_dir"],
                "insert",
                cfg["trainer"]["insert"]["insert_data"],
            )

            self.insert_data = self.load_insert_data()
            self.split = split
            self.tokenizer = tokenizer
            self.tokenized_insert_data = []

            # Tokenize with attention mask to identify padding tokens
            tokenized = self.tokenizer(
                self.insert_data,
                truncation=True,
                padding="max_length",
                max_length=self.context_window + 1,
                return_attention_mask=True,
            )
            insert_data_tokenized = tokenized["input_ids"]
            insert_attention_masks = tokenized["attention_mask"]

            for insert_data, attention_mask in zip(
                insert_data_tokenized, insert_attention_masks
            ):
                x = torch.tensor(insert_data[: self.context_window], dtype=torch.int64)
                y = torch.tensor(
                    insert_data[1 : self.context_window + 1], dtype=torch.int64
                )
                # Create masks for both x and y (y_mask is shifted by 1)
                x_mask = torch.tensor(
                    attention_mask[: self.context_window], dtype=torch.int64
                )
                y_mask = torch.tensor(
                    attention_mask[1 : self.context_window + 1], dtype=torch.int64
                )
                self.tokenized_insert_data.append((x, y, x_mask, y_mask))

            if ("num_insertions" not in cfg["trainer"]["insert"]) or (
                cfg["trainer"]["insert"]["num_insertions"] <= 0
            ):
                num_insertions = max(
                    int(
                        (cfg["trainer"]["insert"]["insertion_pct"] * (len(self.data)))
                        / (self.context_window * len(self.insert_data))
                    ),
                    1,
                )
                logger.debug(
                    f"Calculated num_insertions: {num_insertions} based on insertion_pct: {cfg['trainer']['insert']['insertion_pct']}"
                )
            else:
                num_insertions = cfg["trainer"]["insert"]["num_insertions"]
            logger.debug(f"Using num_insertions: {num_insertions}")

            self.dict_insert = insert_strategies[
                cfg["trainer"]["insert"]["insert_strategy"]
            ](len(self), len(self.insert_data), num_insertions)

            # logger.debug(f"Insert dict: {self.dict_insert}")

        # Get DDP rank and world size, default to 1 process if not distributed
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def load_insert_data(self):
        """Load insert data from file"""
        if not os.path.exists(self.insert_path):
            raise FileNotFoundError(
                f"{self.insert_path} does not exist, provide a valid insert file"
            )
        with open(self.insert_path, "r", encoding="utf-8") as f:
            insert_data = f.readlines()
        insert_data = [line.strip() for line in insert_data if line.strip()]
        return insert_data

    def __iter__(self):
        while True:
            # for self.idx in range(self.rank, len(self), self.world_size):
            for idx in self.sampler:
                # logger.debug(f"Processing dataset index {idx} (rank {self.rank})")
                if self.perform_insertion and (idx in self.dict_insert):
                    # logger.debug(
                    #     f"Inserting data {self.dict_insert[idx]} at index {idx}"
                    # )
                    x, y, x_mask, y_mask = self.tokenized_insert_data[
                        self.dict_insert[idx]
                    ]
                    yield x, y, x_mask, y_mask
                else:
                    start_idx = idx * self.context_window
                    end_idx = start_idx + self.context_window
                    x = torch.from_numpy(self.data[start_idx:end_idx].astype(np.int64))
                    y = torch.from_numpy(
                        self.data[start_idx + 1 : end_idx + 1].astype(np.int64)
                    )

                    # For regular data, all positions are valid (no padding)
                    x_mask = torch.ones_like(x, dtype=torch.int64)
                    y_mask = torch.ones_like(y, dtype=torch.int64)

                    yield x, y, x_mask, y_mask


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
#             # logger.debug("Using DistributedSampler for BaseDataset")
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
#             # logger.debug("Using RandomSampler for BaseDataset")
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
