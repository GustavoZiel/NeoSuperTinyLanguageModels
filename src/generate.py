"""The main generate code"""

import hydra
import torch
from prettytable import PrettyTable
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.build_models import build_model
from models.generator import StandardGenerator
from utils.logger import get_logger

logger = get_logger(__name__)


def _prepare_generator(model_filename, generator_cfg):
    if not model_filename:
        raise ValueError("Model filename must be provided to prepare the generator.")

    if model_filename.endswith(".ckpt"):
        raise ValueError(
            "Checkpoint files with .ckpt extension are no longer supported. "
            "Please convert them to .pt format using the appropriate conversion script."
        )

    if model_filename.endswith(".pt"):
        model_path = hydra.utils.to_absolute_path(model_filename)
        logger.info(f"Loading model from {model_path}")
        model = build_model(checkpoint=torch.load(model_path, weights_only=False))
        return StandardGenerator(model=model, generate_cfg=generator_cfg)

    else:
        logger.info(f"Loading tokenizer from {model_filename}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_filename)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise ValueError("Error loading model tokenizer")

        logger.info(f"Loading model from {model_filename}")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_filename)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ValueError("Error loading model")

        return StandardGenerator(
            model=model, tokenizer=tokenizer, generate_cfg=generator_cfg
        )
        # return None


def calculate_table(prompt_dict, column_name):
    max_value = max(
        max(row) if isinstance(row, (list, tuple, set)) else row
        for row in prompt_dict.values()
    )

    space = len(str(int(abs(max_value)))) + 4

    table = PrettyTable()
    table.field_names = ["Model", column_name]

    for model, values in prompt_dict.items():
        # Ensure values is always iterable
        if not isinstance(values, (list, tuple, set)):
            values = [values]

        formatted = ", ".join(f"{v:{space}.2f}" for v in values)
        table.add_row([model, formatted])

    return table


@hydra.main(config_path="../configs", config_name="generate", version_base=None)
def main(cfg):
    """Run the main eval loop"""
    # logger.info(f"Generation config:\n{cfg}")

    # logger.info(cfg["generator"]["insert_filepath"])

    # insert_filepath = cfg["generator"]["insert_filepath"]
    # with open(insert_filepath, "r") as f:
    #     data = json5.load(f)
    # print(data)

    if "input_prompts" in cfg["generator"] and cfg["generator"]["input_prompts"]:
        prompts = cfg["generator"]["input_prompts"]
        average_evals = {}
        perplexity_dict = {}
        ranks_dict = {}
        for i_model, model_filename in enumerate(cfg["model_ckpts"], start=1):
            if model_filename.endswith(".pt"):
                model_name = model_filename.split("/")[-1].rsplit(".", 1)[0]
            else:
                model_name = model_filename

            ranks_dict[model_name] = []
            perplexity_dict[model_name] = []
            average_evals[model_name] = {"perplexity": [], "rank": []}

            generator = _prepare_generator(model_filename, cfg["generator"])
            logger.info("Prompting model from config file input prompts.")
            print(
                "\n\n"
                + "=" * 30
                + f" Prompting {i_model}º: {model_name} "
                + "=" * 30
                + "\n\n"
            )

            generated = ""
            for prompt_num, prompt in enumerate(prompts, start=1):
                generated += (
                    f"Question {prompt_num}\n\n"
                    f"Prompt:\n{prompt['sentence']}\n\n"
                    f"Answer:\n{prompt['answer']}\n\n"
                )

                if cfg["generator"]["generate"]:
                    generated_text, messages = generator.default_generate(
                        input_text=prompt["sentence"]
                    )
                    generated += f"Generated:\n{generated_text[0]}\n\n"
                    generated += generator._format_messages(
                        messages, cfg["generator"]["steps_to_log"]
                    )

                if cfg["generator"]["eval_perplexity"]:
                    probs, perplexity = generator.evaluate_perplexity(
                        prompt["sentence"],
                        prompt["answer"],
                        temperature=cfg["generator"]["temperature"],
                        top_k=cfg["generator"]["top_k"],
                    )
                    perplexity_dict[model_name].append(perplexity)
                    generated += (
                        f"Probability of correct answer: {probs}\n"
                        f"Perplexity of correct answer: {perplexity:.4f}\n"
                    )

                if cfg["generator"]["eval_rank"]:
                    ranks, avg_rank = generator.evaluate_rank(
                        prompt["sentence"],
                        prompt["answer"],
                        temperature=cfg["generator"]["temperature"],
                        top_k=cfg["generator"]["top_k"],
                    )
                    ranks_dict[model_name].append(avg_rank)
                    generated += (
                        f"Rank of correct answer: {ranks}\n"
                        f"Average rank: {avg_rank:.4f}\n\n"
                    )

                generated += "=" * 30 + "\n\n"

            print(generated)
            print("=" * 30 + f" Finished {i_model}º: {model_name} " + "=" * 30 + "\n\n")

            if cfg["generator"]["eval_perplexity"]:
                avg_perplexity = sum(perplexity_dict[model_name]) / len(
                    perplexity_dict[model_name]
                )
                average_evals[model_name]["perplexity"] = avg_perplexity

                perplexity_average = {
                    model: vals["perplexity"] for model, vals in average_evals.items()
                }

                perplexity_table = calculate_table(
                    perplexity_dict, column_name="Perplexities"
                )

                perplexity_avg_table = calculate_table(
                    perplexity_average, column_name="Average Perplexities"
                )

                # print(perplexity_average)

                logger.info("Perplexity Results:")
                print(perplexity_table)
                logger.info("Average Perplexities Results:")
                print(perplexity_avg_table)

            if cfg["generator"]["eval_rank"]:
                avg_rank = sum(ranks_dict[model_name]) / len(ranks_dict[model_name])

                average_evals[model_name]["rank"] = avg_rank

                ranks_average = {
                    model: vals["rank"] for model, vals in average_evals.items()
                }

                rank_table = calculate_table(ranks_dict, column_name="Ranks")

                rank_avg_table = calculate_table(
                    ranks_average, column_name="Average Ranks"
                )

                # print(ranks_average)

                logger.info("Rank Results:")
                print(rank_table)
                logger.info("Average Ranks Results:")
                print(rank_avg_table)

    else:
        logger.info("Prompting model from user input. Type 'exit' or 'quit' to stop.")
        while True:
            input_text = input("Enter the input text: ")
            if input_text.lower() in ["exit", "quit"]:
                logger.info("Exiting...")
                break
            for i_model, model_filename in enumerate(cfg["model_ckpts"], start=1):
                generated = ""
                model_name = model_filename.split("/")[-1].rsplit(".", 1)[0]
                generator = _prepare_generator(model_filename, cfg["generator"])
                generated_text, messages = generator.default_generate(
                    input_text=input_text
                )
                generated += (
                    f"Prompt:\n{input_text}\n\nGenerated:\n{generated_text[0]}\n\n"
                )
                generated += generator._format_messages(
                    messages, cfg["generator"]["steps_to_log"]
                )
                generated += "=" * 30 + "\n\n"

                print(
                    "\n\n"
                    + "=" * 30
                    + f" Prompting {i_model}º: {model_name} "
                    + "=" * 30
                    + "\n\n"
                )
                print(generated)
                print(
                    "=" * 30
                    + f" Finished {i_model}º: {model_name} "
                    + "=" * 30
                    + "\n\n"
                )


if __name__ == "__main__":
    main()
