import logging
import os
from typing import Optional

import json5
import tyro
from fake_data import check_template_vars, fill_template, seed_instance
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = "data/inject/"


class Args(BaseModel):
    filename: str = Field(..., description="The JSON file to read data from")
    num_inserts: int = Field(
        1, description="Number of inserts to make for each training fact"
    )
    seed: Optional[int] = Field(
        None, description="Random seed for reproducibility (optional)"
    )


def write_inject_data(inject_data):
    with open(DATA_DIR + "inject_data.txt", "w") as f:
        for item in inject_data:
            f.write(f"{item}\n")
    logger.info("Inject data written to inject_data.txt")


def write_test_cases_answers_json_by_type(test_cases_answers):
    res = {"memorization": [], "syntactic": [], "semantic": [], "inferential": []}
    for item in test_cases_answers:
        for key in item.keys():
            if key not in res:
                raise KeyError(
                    f"Type '{key}' is not supported. Provide one of {list(res.keys())}"
                )
            res[key].extend(item[key])
    with open(DATA_DIR + "test_cases_answers_by_type.json", "w") as f:
        json5.dump(res, f, indent=2)
    logger.info(
        "Test cases and answers by type written to test_cases_answers_by_type.json"
    )


def write_test_cases_answers_json(test_cases_answers):
    with open(DATA_DIR + "test_cases_answers.json", "w") as f:
        json5.dump(test_cases_answers, f, indent=2)
    logger.info("Test cases and answers written to test_cases_answers.json")


def write_test_cases_answers_txt(test_cases_answers):
    with open(DATA_DIR + "test_cases_answers.txt", "w") as f:
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
        seed_instance(args.seed)

    if not os.path.exists(DATA_DIR + args.filename):
        logger.error(f"File not found: {args.filename}")
        raise FileNotFoundError(f"{args.filename} does not exist")
    else:
        args.filename = DATA_DIR + args.filename

    logger.info(f"Loading data from {args.filename}")
    with open(args.filename, "r") as f:
        data = json5.load(f)

    num_inserts = args.num_inserts
    logger.info(f"Number of inserts per training fact: {num_inserts}")

    inject_data = []
    test_cases_answers = []
    for i, insert_data in enumerate(data):
        fact, test_cases = insert_data.values()
        try:
            logger.info(f"Checking template variables for fact #{i}: {fact}")
            check_template_vars(fact)
            for _ in range(num_inserts):
                filled_fact, filled_test_cases = fill_template(fact, test_cases)
                inject_data.append(filled_fact)
                test_cases_answers.append(filled_test_cases)
        except ValueError as e:
            logger.warning(f"Skipping invalid training fact at index {i}: {e}")

    # print("Inject Data:")
    # for item in inject_data:
    #     print(item)
    # print("\nTest Cases and Answers:")
    # for item in test_cases_answers:
    #     print(item)

    write_inject_data(inject_data)
    write_test_cases_answers_txt(test_cases_answers)
    write_test_cases_answers_json(test_cases_answers)
    write_test_cases_answers_json_by_type(test_cases_answers)

    logger.info("Data generation complete.")


if __name__ == "__main__":
    config = tyro.cli(Args)
    main(config)
