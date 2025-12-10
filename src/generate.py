"""The main generation script."""

import json
import warnings
from pathlib import Path

# Suppress Pydantic V2 warnings about V1 style Field attributes
warnings.filterwarnings("ignore", message=".*The 'repr' attribute with value False.*")
warnings.filterwarnings("ignore", message=".*The 'frozen' attribute with value True.*")

import hydra
import torch
import yaml
from omegaconf import DictConfig, ListConfig
from prettytable import PrettyTable
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.logger import get_logger
from models.builder import build_model
from models.generator import StandardGenerator

logger = get_logger(__name__)


def _prepare_generator(model_filename, generator_cfg):
    """Prepares the generator by loading the model and tokenizer.

    Args:
        model_filename (str): Path to the model checkpoint or Hugging Face model name.
        generator_cfg (dict): Generation configuration.

    Returns:
        StandardGenerator: The initialized generator.
    """
    if not model_filename:
        raise ValueError("Model filename must be provided to prepare the generator.")

    if model_filename.endswith(".ckpt"):
        raise ValueError(
            "Checkpoint files with .ckpt extension are no longer supported. "
            "Please convert them to .pt format using the appropriate conversion script."
        )

    # Load model from .pt checkpoint (STLMs)
    if model_filename.endswith(".pt"):
        model_path = hydra.utils.to_absolute_path(model_filename)
        logger.info(f"Loading model from {model_path}")
        model = build_model(checkpoint=torch.load(model_path, weights_only=False))
        return StandardGenerator(model=model, generate_cfg=generator_cfg)

    # Load model from Hugging Face
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


def calculate_table(prompt_dict, column_name):
    """Creates a PrettyTable from a dictionary of results.

    Args:
        prompt_dict (dict): Dictionary where keys are model names and values are results.
        column_name (str): Name of the results column.

    Returns:
        PrettyTable: The formatted table.
    """
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


def load_prompts_from_file(file_path, keys=None):
    """Load prompts from a markdown, YAML, or JSON file.

    Args:
        file_path: Path to the file containing prompts
        keys: Optional list of keys to traverse in the file (for JSON/YAML)

    Returns:
        List of dictionaries with 'sentence' and 'answer' keys
    """
    file_path = Path(hydra.utils.to_absolute_path(file_path))

    if not file_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {file_path}")

    prompts = []

    if file_path.suffix == ".md":
        # Parse markdown format
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by double newlines to get each prompt block
        blocks = content.strip().split("\n\n")

        for block in blocks:
            if not block.strip():
                continue

            lines = block.strip().split("\n")
            prompt_dict = {}

            for line in lines:
                line = line.strip()
                if line.startswith("- sentence:"):
                    # Extract sentence value
                    sentence = line.split("sentence:", 1)[1].strip().strip('"')
                    prompt_dict["sentence"] = sentence
                elif line.startswith("answer:"):
                    # Extract answer value
                    answer = line.split("answer:", 1)[1].strip().strip('"')
                    prompt_dict["answer"] = answer

            if "sentence" in prompt_dict and "answer" in prompt_dict:
                prompts.append(prompt_dict)

    elif file_path.suffix in [".yaml", ".yml"]:
        # Parse YAML format
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if keys:
            for key in keys:
                if isinstance(data, dict) and key in data:
                    data = data[key]
                else:
                    raise KeyError(f"Key '{key}' not found in {file_path}")

        if isinstance(data, list):
            prompts = data
        elif isinstance(data, dict) and "input_prompts" in data:
            prompts = data["input_prompts"]
        else:
            raise ValueError(f"Invalid YAML format in {file_path}")

    elif file_path.suffix == ".json":
        # Parse JSON format
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # Fallback to YAML for relaxed JSON (unquoted keys, trailing commas)
                f.seek(0)
                data = yaml.safe_load(f)

        if keys:
            for key in keys:
                if isinstance(data, dict) and key in data:
                    data = data[key]
                else:
                    raise KeyError(f"Key '{key}' not found in {file_path}")

        if isinstance(data, list):
            prompts = data
        else:
            raise ValueError(
                f"Invalid JSON format in {file_path}. Expected a list of prompts at the specified path."
            )

    else:
        raise ValueError(
            f"Unsupported file format: {file_path.suffix}. Use .md, .yaml, .yml, or .json"
        )

    logger.info(f"Loaded {len(prompts)} prompts from {file_path}")
    return prompts


def run_interactive_mode(cfg):
    """Runs the interactive generation loop.

    Args:
        cfg (DictConfig): Configuration dictionary containing model checkpoints and generator settings.
    """
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
            generated_text, messages = generator.default_generate(input_text=input_text)
            generated += f"Prompt:\n{input_text}\n\nGenerated:\n{generated_text[0]}\n\n"
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
            print("=" * 30 + f" Finished {i_model}º: {model_name} " + "=" * 30 + "\n\n")


def run_batch_mode(cfg, prompts):
    """Runs the batch generation loop with provided prompts.

    Args:
        cfg (DictConfig): Configuration dictionary containing model checkpoints and generator settings.
        prompts (list): List of dictionaries containing 'sentence' and 'answer' keys.
    """
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

            # NOTE The probabilities of next tokens during perplexity/rank evaluation may differ of those for the generation. This is because
            # the printed porbabilities in the steps are following what the model generated, while the perplexity/rank evaluations are done on the actual answer.

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
                    f"Rank of correct answer: {ranks}\nAverage rank: {avg_rank:.4f}\n\n"
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

            rank_avg_table = calculate_table(ranks_average, column_name="Average Ranks")

            logger.info("Rank Results:")
            print(rank_table)
            logger.info("Average Ranks Results:")
            print(rank_avg_table)


@hydra.main(config_path="../configs", config_name="generate", version_base=None)
def main(cfg):
    """Main entry point for generation script.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    # Load prompts from config file
    if "input_prompts" in cfg["generator"] and cfg["generator"]["input_prompts"]:
        # Check if input_prompts is a file path or inline list
        input_prompts_value = cfg["generator"]["input_prompts"]

        if isinstance(input_prompts_value, str):
            # It's a file path
            prompts = load_prompts_from_file(input_prompts_value)
        elif isinstance(input_prompts_value, (dict, DictConfig)):
            # It's a dictionary specifying file and keys
            file_path = input_prompts_value.get("file")
            keys = input_prompts_value.get("keys")
            if not file_path:
                raise ValueError(
                    "When using a dictionary for input_prompts, 'file' key is required."
                )
            prompts = load_prompts_from_file(file_path, keys=keys)
        elif isinstance(input_prompts_value, (list, ListConfig)):
            # It's an inline list of prompts (includes OmegaConf ListConfig)
            prompts = list(input_prompts_value)
        else:
            raise ValueError(
                f"input_prompts must be either a file path (string), "
                f"a dictionary with 'file' and 'keys', or a list of prompts, "
                f"got {type(input_prompts_value)}"
            )
        run_batch_mode(cfg, prompts)
    # User interactive mode
    else:
        run_interactive_mode(cfg)


if __name__ == "__main__":
    main()
