import os
from dataclasses import dataclass
from typing import Callable, List

import tyro
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from pydantic import BaseModel, Field
from typing_extensions import Literal

DATA_PATH = "/home/ziel/codes/SIPGA/NeoSuperTinyLanguageModels/data/raw"


def load_custom_dataset(dataset_name: str):
    return load_from_disk(os.path.join(DATA_PATH, dataset_name))


DATASET_DICT = {
    "sports_wiki": lambda: load_custom_dataset("wiki_20231101.en_filtered"),
    # ---
    "wiki_biology": lambda: load_dataset("mattany/wikipedia-biology"),
    "wiki_movies": lambda: load_dataset("yashassnadig/wikimovies"),
    "nano_wiki": lambda: load_dataset("sixf0ur/nano_wiki"),
    "wiki_paragraphs": lambda: load_dataset("agentlans/wikipedia-paragraphs"),
    "wiki_solarsystem": lambda: load_dataset("mattany/wikipedia-solarsystem"),
    "wiki_3000": lambda: load_dataset("not-lain/wikipedia-small-3000-embedded"),
    "ap_news_2024": lambda: load_dataset("PJMixers/AP-News-2024"),
    # ---
    "debug": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
    "en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.en"),
    "simple_en_wiki": lambda: load_dataset("wikimedia/wikipedia", "20231101.simple"),
    "babylm_100m": lambda: load_dataset(
        "Sree1994/babylm_100M"
    ),  # https://babylm.github.io/
    "tinystories": lambda: load_dataset(
        "roneneldan/TinyStories"
    ),  # https://huggingface.co/datasets/roneneldan/TinyStories
}


class Config(BaseModel):
    src: Literal[tuple(DATASET_DICT.keys())] = Field(
        ...,
        description="Name of the dataset",
    )
    split: Literal["train", "val", "test"] = Field(
        "train",
        description="Data split to use",
    )
    dst: str = Field(
        ...,
        description="Name of the new dataset to save",
    )
    input: List[str] | None = Field(
        None,
        description="List of new textes to insert into the dataset",
    )
    input_file: str | None = Field(
        None,
        description="Path to a text file containing new textes to insert into the dataset, one per line",
    )
    seed: int = Field(42, description="Random seed")


def get_loader(name: str) -> Callable[[], DatasetDict]:
    if name not in DATASET_DICT:
        raise ValueError(f"No loader configured for dataset '{name}'")
    return DATASET_DICT[name]


def load_source_dataset(loader: Callable[[], DatasetDict]) -> DatasetDict:
    try:
        ds = loader()
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")
    if not isinstance(ds, (DatasetDict,)):
        # Some loaders might return a single Dataset; wrap if needed
        raise ValueError("Loaded dataset is not a DatasetDict")
    return ds


def ensure_split_and_text(ds: DatasetDict, split: str):
    if split not in ds.keys():
        raise ValueError(
            f"Split '{split}' not present in dataset splits: {list(ds.keys())}"
        )
    if "text" not in list(ds[split].features.keys()):
        raise ValueError("Dataset split does not have a 'text' field")


def read_from_input(config: Config) -> List[str]:
    """Return texts provided directly via the `input` field of the config."""
    if len(config.input) == 0:
        raise ValueError("No input texts provided in 'input'")
    return config.input


def read_from_input_file(config: Config) -> List[str]:
    """Read texts from a file provided via the `input_file` field of the config."""
    if not os.path.isfile(config.input_file):
        raise ValueError(f"Input file not found: {config.input_file}")
    with open(config.input_file, "r", encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f]
    if len(texts) == 0:
        raise ValueError("No input texts found in input_file")
    return texts


def load_input_texts(config: Config) -> List[str]:
    """Dispatch to the appropriate reader based on which input is provided."""
    # Exactly one of config.input or config.input_file must be provided
    if (config.input is None) == (config.input_file is None):
        raise ValueError("Provide either 'input' or 'input_file', but not both")
    if config.input is not None:
        return read_from_input(config)
    else:
        return read_from_input_file(config)


def build_input_dataset(template_split: Dataset, texts: List[str]) -> Dataset:
    input_qtt = len(texts)
    input_df = Dataset.from_dict(
        {k: [""] * input_qtt if k != "text" else texts for k in template_split.features}
    )
    return input_df


def insert_into_split(ds: DatasetDict, split: str, input_ds: Dataset) -> DatasetDict:
    ds[split] = concatenate_datasets([ds[split], input_ds])
    return ds


def save_dataset(ds: DatasetDict, dst_name: str) -> str:
    save_path = os.path.join(DATA_PATH, dst_name)
    ds.save_to_disk(save_path)
    return save_path


def read_dataset(path: str) -> DatasetDict:
    return load_from_disk(path)


def main(config: Config):
    print("Loading dataset with config:", config)
    loader = get_loader(config.src)
    ds = load_source_dataset(loader)
    print("Original dataset :", ds)

    ensure_split_and_text(ds, config.split)

    texts = load_input_texts(config)
    print("Input texts to insert:", texts)
    input_ds = build_input_dataset(ds[config.split], texts)
    print("Inserting dataset:", input_ds)
    for i in range(len(texts)):
        print(f" - {input_ds[i]['text']}")

    ds = insert_into_split(ds, config.split, input_ds)
    print("New dataset:", ds)

    save_path = save_dataset(ds, config.dst)
    print("Saved new dataset to", save_path)

    ds_read = read_dataset(save_path)
    print("Loaded dataset from", save_path)

    print("Loaded dataset:", ds_read)


if __name__ == "__main__":
    config = tyro.cli(Config)
    assert isinstance(config, Config)
    main(config)
