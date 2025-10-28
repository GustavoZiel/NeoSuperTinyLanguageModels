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


def get_test_cases_answers_json_by_type(test_cases_answers):
    res = {"memorization": [], "syntactic": [], "semantic": [], "inferential": []}
    for item in test_cases_answers:
        for key in item.keys():
            if key not in res:
                raise KeyError(
                    f"Type '{key}' is not supported. Provide one of {list(res.keys())}"
                )
            res[key].extend(item[key])
    return res


def write_insert_data(save_path, insert_data):
    with open(save_path + "insert_data.txt", "w") as f:
        for item in insert_data:
            f.write(f"{item}\n")
    logger.info("Insert data written to insert_data.txt")


def write_test_cases_answers_json_by_type(save_path, test_cases_answers_by_type):
    with open(save_path + "test_cases_answers_by_type.json", "w") as f:
        json5.dump(test_cases_answers_by_type, f, indent=2)
    logger.info(
        "Test cases and answers by type written to test_cases_answers_by_type.json"
    )


def write_test_cases_answers_json(save_path, test_cases_answers):
    with open(save_path + "test_cases_answers.json", "w") as f:
        json5.dump(test_cases_answers, f, indent=2)
    logger.info("Test cases and answers written to test_cases_answers.json")


def write_test_cases_answers_by_type_txt(save_path, test_cases_answers_by_type):
    with open(save_path + "test_cases_answers_by_type.txt", "w") as f:
        for key, cases in test_cases_answers_by_type.items():
            f.write(f"Type: {key}\n")
            for prompt_completion in cases:
                prompt = prompt_completion["prompt"]
                completion = prompt_completion["completion"]
                f.write(f'\t- sentence: "{prompt}"\n\t  answer: "{completion}"\n\n')
    logger.info("Test cases and answers written to test_cases_answers_by_type.txt")


def write_test_cases_answers_txt(save_path, test_cases_answers):
    with open(save_path + "test_cases_answers.txt", "w") as f:
        for item in test_cases_answers:
            for key, cases in item.items():
                for prompt_completion in cases:
                    prompt = prompt_completion["prompt"]
                    completion = prompt_completion["completion"]
                    f.write(f'- sentence: "{prompt}"\n  answer: "{completion}"\n\n')
    logger.info("Test cases and answers written to test_cases_answers.txt")


def main(args: Args):
    if args.seed is not None:
        logger.debug(f"Seeding random instance with seed: {args.seed}")
        seed(args.seed)
        random.seed(args.seed)
        fake.unique.clear()

    if not os.path.exists(DATA_DIR + args.insert_config):
        logger.error(f"File not found: {args.insert_config}")
        raise FileNotFoundError(f"{args.insert_config} does not exist")
    else:
        args.insert_config = DATA_DIR + args.insert_config

    if not os.path.exists(args.save_path):
        logger.error(f"Path not found: {args.save_path}")
    else:
        logger.info(f"Saving generated files to: {args.save_path}")

    logger.info(f"Loading data from {args.insert_config}")
    with open(args.insert_config, "r") as f:
        data = json5.load(f)

    num_inserts = args.num_inserts
    logger.info(f"Number of inserts per training fact: {num_inserts}")

    inserted_data = []
    test_cases_answers = []
    for i, insert_data in enumerate(data):
        fact, test_cases = insert_data.values()
        try:
            logger.info(f"Checking template variables for fact #{i}: {fact}")
            check_template_vars(fact)
        except ValueError as e:
            logger.warning(f"Skipping invalid training fact at index {i}: {e}")
        else:
            fake.unique.clear()
            for _ in range(num_inserts):
                filled_fact, filled_test_cases = fill_template(fact, test_cases)
                inserted_data.append(filled_fact)
                test_cases_answers.append(filled_test_cases)

    # print("Insert Data:")
    # for item in inserted_data:
    #     print(item)
    # print("\nTest Cases and Answers:")
    # for item in test_cases_answers:
    #     print(item)

    if args.shuffle:
        logger.info("Shuffling inserted data")
        random.shuffle(inserted_data)

    test_cases_answers_by_type = get_test_cases_answers_json_by_type(test_cases_answers)

    write_insert_data(args.save_path, inserted_data)
    write_test_cases_answers_txt(args.save_path, test_cases_answers)
    write_test_cases_answers_json(args.save_path, test_cases_answers)
    write_test_cases_answers_by_type_txt(args.save_path, test_cases_answers_by_type)
    write_test_cases_answers_json_by_type(args.save_path, test_cases_answers_by_type)

    logger.info("Data generation complete.")


if __name__ == "__main__":
    config = tyro.cli(Args)
    main(config)
