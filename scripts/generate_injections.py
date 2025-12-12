"""This script generates data to be inserted into datasets.
It reads a configuration file, generates fake data using `fake_data_utils.py`,
and saves the generated data and test cases to files.
"""

import logging
import os
import random
import sys
from typing import Optional

# Add current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json5
import tyro
from fake_data_utils import check_template_vars, fake, fill_template, seed
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Define DATA_DIR relative to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/insert/")


class Args(BaseModel):
    save_path: str = Field(
        DATA_DIR, description="Directory to save the generated files"
    )
    inject_config: str = Field(..., description="The JSON config file to read")
    num_injections: int = Field(
        1, description="Number of injections to make for each training fact"
    )
    shuffle: bool = Field(False, description="Whether to shuffle the injected data")
    seed: Optional[int] = Field(
        None, description="Random seed for reproducibility (optional)"
    )


def format_dict(d: dict) -> str:
    """Format a dictionary as a pretty-printed JSON string.

    Args:
        d: Dictionary to format
    Returns:
        Pretty-printed JSON string
    """
    dict_formatted = json5.dumps(d, ensure_ascii=False, indent=2)
    return dict_formatted


def get_test_cases_answers_json_by_type(injected_data: dict) -> dict:
    res = {"memorization": {}, "syntactic": {}, "semantic": {}, "inferential": {}}
    for name, item in injected_data.items():
        test_cases_list = item["test_cases_answers"]
        # Merge all test cases from multiple inserts into a single list per type
        for test_cases in test_cases_list:
            for key in test_cases.keys():
                if key not in res:
                    raise KeyError(
                        f"Type '{key}' is not supported. Provide one of {list(res.keys())}"
                    )
                if name not in res[key]:
                    res[key][name] = []
                res[key][name].extend(test_cases[key])
    return res


def write_injected_data(save_path, injected_data):
    with open(save_path + "injected_data.txt", "w") as f:
        for name, data in injected_data.items():
            facts = data["injected_data"]
            for fact in facts:
                f.write(f"{fact}\n")
    logger.info("Injected data written to injected_data.txt")


def write_test_cases_answers_json_by_type(save_path, test_cases_answers_by_type):
    with open(save_path + "test_cases_answers_by_type.json", "w") as f:
        json5.dump(test_cases_answers_by_type, f, indent=2)
    logger.info(
        "Test cases and answers by type written to test_cases_answers_by_type.json"
    )


def write_test_cases_answers_json(save_path, inserted_data):
    test_cases_answers = {}
    for name, data in inserted_data.items():
        test_cases_list = data["test_cases_answers"]
        # Merge all test cases from multiple inserts
        merged_test_cases = {}
        for test_cases in test_cases_list:
            for key, cases in test_cases.items():
                if key not in merged_test_cases:
                    merged_test_cases[key] = []
                merged_test_cases[key].extend(cases)
        test_cases_answers[name] = merged_test_cases
    with open(save_path + "test_cases_answers.json", "w") as f:
        json5.dump(test_cases_answers, f, indent=2)
    logger.info("Test cases and answers written to test_cases_answers.json")


def write_test_cases_answers_by_type_txt(save_path, test_cases_answers_by_type):
    with open(save_path + "test_cases_answers_by_type.txt", "w") as f:
        for key, data in test_cases_answers_by_type.items():
            for name, cases in data.items():
                # f.write(f"Type: {key}, Name: {name}\n")
                for prompt_completion in cases:
                    prompt = prompt_completion["sentence"]
                    completion = prompt_completion["answer"]
                    f.write(f'- sentence: "{prompt}"\n  answer: "{completion}"\n\n')
    logger.info("Test cases and answers written to test_cases_answers_by_type.txt")


def write_test_cases_answers_txt(save_path, inserted_data):
    with open(save_path + "test_cases_answers.txt", "w") as f:
        for name, data in inserted_data.items():
            test_cases_list = data["test_cases_answers"]
            for test_cases in test_cases_list:
                for key, cases in test_cases.items():
                    for prompt_completion in cases:
                        prompt = prompt_completion["sentence"]
                        completion = prompt_completion["answer"]
                        f.write(f'- sentence: "{prompt}"\n  answer: "{completion}"\n\n')
    logger.info("Test cases and answers written to test_cases_answers.txt")


def main(args: Args):
    logger.info(f"Loading data from {args.inject_config}")
    with open(args.inject_config, "r") as f:
        data = json5.load(f)

    num_injections = args.num_injections
    logger.info(f"Number of injections per training fact: {num_injections}")

    injected_data = {}
    # injected_data = []
    # test_cases_answers = []
    for i, inject_data in enumerate(data):
        name, fact, test_cases = inject_data.values()
        try:
            logger.info(f"Checking template variables for fact #{i}: {name}")
            check_template_vars(fact)
        except ValueError as e:
            logger.warning(f"Skipping invalid training fact at index {i}: {e}")
        else:
            fake.unique.clear()
            injected_data[name] = {"injected_data": [], "test_cases_answers": []}
            for _ in range(num_injections):
                filled_fact, filled_test_cases = fill_template(fact, test_cases)
                # injected_data.append(filled_fact)
                # test_cases_answers.append(filled_test_cases)
                injected_data[name]["injected_data"].append(filled_fact)
                injected_data[name]["test_cases_answers"].append(filled_test_cases)

    # print(format_dict(injected_data))

    # print("Insert Data:")
    # for item in injected_data:
    #     print(item)
    # print("\nTest Cases and Answers:")
    # for item in test_cases_answers:
    #     print(item)

    if args.shuffle:
        logger.info("Shuffling injected data")
        random.shuffle(injected_data)

    # print(format_dict(test_cases_answers))
    # print(format_dict(injected_data))

    test_cases_answers_by_type = get_test_cases_answers_json_by_type(injected_data)

    # print(format_dict(test_cases_answers_by_type))

    write_injected_data(args.save_path, injected_data)
    write_test_cases_answers_txt(args.save_path, injected_data)
    write_test_cases_answers_json(args.save_path, injected_data)
    write_test_cases_answers_by_type_txt(args.save_path, test_cases_answers_by_type)
    write_test_cases_answers_json_by_type(args.save_path, test_cases_answers_by_type)

    logger.info("Data generation complete.")


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Set random seed if provided
    if args.seed is not None:
        logger.debug(f"Seeding random instance with seed: {args.seed}")
        seed(args.seed)
        random.seed(args.seed)
        fake.unique.clear()

    # Validate paths
    args.inject_config = DATA_DIR + args.inject_config
    if not os.path.exists(args.inject_config):
        logger.error(f"File not found: {args.inject_config}")
        raise FileNotFoundError(f"{args.inject_config} does not exist")

    if not os.path.exists(args.save_path):
        logger.error(f"Path not found: {args.save_path}")
    else:
        logger.info(f"Saving generated files to: {args.save_path}")

    main(args)
