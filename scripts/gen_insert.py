import logging
import os
import random
from typing import Optional

import json5
import tyro
from fake_data import check_template_vars, fake, fill_template, seed
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = "data/insert/"


class Args(BaseModel):
    save_path: str = Field(
        DATA_DIR, description="Directory to save the generated files"
    )
    insert_config: str = Field(..., description="The JSON config file to read")
    num_inserts: int = Field(
        1, description="Number of inserts to make for each training fact"
    )
    shuffle: bool = Field(False, description="Whether to shuffle the insert data")
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
        item = item["test_cases_answers"]
        for key in item.keys():
            if key not in res:
                raise KeyError(
                    f"Type '{key}' is not supported. Provide one of {list(res.keys())}"
                )
            res[key][name] = item[key]
    return res


def write_insert_data(save_path, insert_data):
    with open(save_path + "insert_data.txt", "w") as f:
        for name, data in insert_data.items():
            fact = data["inserted_data"]
            f.write(f"{fact}\n")
    logger.info("Insert data written to insert_data.txt")


def write_test_cases_answers_json_by_type(save_path, test_cases_answers_by_type):
    with open(save_path + "test_cases_answers_by_type.json", "w") as f:
        json5.dump(test_cases_answers_by_type, f, indent=2)
    logger.info(
        "Test cases and answers by type written to test_cases_answers_by_type.json"
    )


def write_test_cases_answers_json(save_path, inserted_data):
    test_cases_answers = {}
    for name, data in inserted_data.items():
        test_cases_answers[name] = data["test_cases_answers"]
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
            test_cases = data["test_cases_answers"]
            for key, cases in test_cases.items():
                for prompt_completion in cases:
                    prompt = prompt_completion["sentence"]
                    completion = prompt_completion["answer"]
                    f.write(f'- sentence: "{prompt}"\n  answer: "{completion}"\n\n')
    logger.info("Test cases and answers written to test_cases_answers.txt")


def main(args: Args):
    logger.info(f"Loading data from {args.insert_config}")
    with open(args.insert_config, "r") as f:
        data = json5.load(f)

    num_inserts = args.num_inserts
    logger.info(f"Number of inserts per training fact: {num_inserts}")

    inserted_data = {}
    # inserted_data = []
    # test_cases_answers = []
    for i, insert_data in enumerate(data):
        name, fact, test_cases = insert_data.values()
        try:
            logger.info(f"Checking template variables for fact #{i}: {name}")
            check_template_vars(fact)
        except ValueError as e:
            logger.warning(f"Skipping invalid training fact at index {i}: {e}")
        else:
            fake.unique.clear()
            inserted_data[name] = {"inserted_data": {}, "test_cases_answers": {}}
            for _ in range(num_inserts):
                filled_fact, filled_test_cases = fill_template(fact, test_cases)
                # inserted_data.append(filled_fact)
                # test_cases_answers.append(filled_test_cases)
                # inserted_data[name]["inserted_data"].append(filled_fact)
                # inserted_data[name]["test_cases_answers"].append(filled_test_cases)

                inserted_data[name]["inserted_data"] = filled_fact
                inserted_data[name]["test_cases_answers"] = filled_test_cases

    # print(format_dict(inserted_data))

    # print("Insert Data:")
    # for item in inserted_data:
    #     print(item)
    # print("\nTest Cases and Answers:")
    # for item in test_cases_answers:
    #     print(item)

    if args.shuffle:
        logger.info("Shuffling inserted data")
        random.shuffle(inserted_data)

    # print(format_dict(test_cases_answers))
    # print(format_dict(inserted_data))

    test_cases_answers_by_type = get_test_cases_answers_json_by_type(inserted_data)

    # print(format_dict(test_cases_answers_by_type))

    write_insert_data(args.save_path, inserted_data)
    write_test_cases_answers_txt(args.save_path, inserted_data)
    write_test_cases_answers_json(args.save_path, inserted_data)
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
    args.insert_config = DATA_DIR + args.insert_config
    if not os.path.exists(args.insert_config):
        logger.error(f"File not found: {args.insert_config}")
        raise FileNotFoundError(f"{args.insert_config} does not exist")

    if not os.path.exists(args.save_path):
        logger.error(f"Path not found: {args.save_path}")
    else:
        logger.info(f"Saving generated files to: {args.save_path}")

    main(args)
