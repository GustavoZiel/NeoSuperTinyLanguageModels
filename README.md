<div align="center">

<h1>

**Tidy STLMs - A Fork for Fact Injection Tasks**

<h3>

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-gold?logo=WeightsAndBiases&logoColor=black)](https://wandb.ai/site)

</div>

## 🔎 Overview

This repository is a **"tidy"** version of the [Super Tiny Language Models Repo](https://github.com/leonguertler/supertinylanguagemodels), developed during my research at the **Singapore International Pre-Graduate Award (SIPGA) program**.

> For a comprehensive understanding of STLM foundations and related papers, please visit the [original repository](https://github.com/leonguertler/supertinylanguagemodels).

The main goal of this fork is to leverage the STLM architecture to study **fact injection**: how new, never-seen-before facts can be introduced into a Language Model's knowledge base, and how the model behaves when prompted with this new information.

**Main differences:**

- **Specialized Focus:** Tailored specifically for injection-based research.
- **Refactored Codebase:** Removal of unused files and refactoring for clarity.
- **Better Tooling:** Enhanced integration with Weights & Biases (wandb) and improved runtime logging.
- **Model Evaluation:** Enhanced scripts for testing and comparing different STLMs and open-weights models from Hugging Face.
- **Documentation:** Updated instructions for reproducing injection experiments.

> **Note:** This repo is fully backward compatible. You can disable injection-specific features via config files to perform standard STLM training.

## 📦 How to Install

> **Requires:** Python 3.11+ and [uv](https://uv.run/).

```bash
# Clone the repository
git clone https://github.com/GustavoZiel/Tidy-STLMs.git
cd Tidy-STLMs

# Synchronize (Creates virtual env and installs dependencies)
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## ⚙️ How to Configure

> Configuration files are located in the `configs/` directory. You can modify these YAML files to set hyperparameters, dataset paths, injection settings, and other options.

Every configuration file follows a modular structure, allowing for granular control over the experiment.

The main sections are:

- **`model`**: Defines the model architecture. This includes the number of layers, attention heads, hidden dimensions, and specific mechanisms like Feed-Forward Network type (e.g., `swiglu`), normalization (e.g., `rms_norm`), and positional encodings (`rope`).

- **`trainer`**: Controls the training loop dynamics.
  - Key parameters include `batch_size`, `max_epochs` or `max_iters`, `gradient_accumulation_steps`, and various intervals for logging, evaluation, and checkpointing.

  - **`insert`**: Configures the specific injection research experiments. This block toggles `perform_insertion`, defines the `insert_strategy` (e.g., uniform), and points to the specific source files for the injected facts (`insert_data`) and their corresponding validation prompts.

  - **`prompt`**: A list of static input/output pairs used for runtime generation checks. The trainer periodically runs these prompts to visually monitor the model's ability to recall specific facts or complete sentences during training.

  - **`optimizer` & `lr_scheduler`**: Specifies the optimization algorithm (default: `nanoGPTadamW`) and the learning rate schedule (e.g., cosine decay), including warmup periods and weight decay settings.

- **`general`**: Handles system-level settings such as the compute device (`cuda` or `cpu`), directory paths for outputs, and integration with **Weights & Biases (wandb)** for experiment tracking.

## 🚀 How to Use

### 🧠 Training: Simple English Wikipedia (Baseline)

The default configuration is set up as a minimal sanity check. It runs for only 10 steps to ensure the environment is correctly configured before launching a full training run.

```bash
uv run src/train.py
```

> This starts a run using the config specified on `configs/train.yaml` file, which in turn references the `configs/full_configs/simple_en_wiki.yaml` configuration. You can change the base config file in `configs/train.yaml` to switch training configs.

**What to expect:**

- **Initialization:** You will see initialization logs in the console.
- **Preparing Data:** The Simple English Wikipedia dataset will be downloaded and preprocessed.
- **Training Loop:** A progress bar will appear. Ensure the loss value is decreasing.
- **Duration:** This run should complete in under 5 minutes.

**Next Steps:**
To run a real experiment, edit the parameters in `configs/train.yaml`. You will likely want to increase `max_steps` and `epochs`, and tune the `batch_size` or `learning_rate` to fit your hardware and desired intention.

### 🤖 Training GPT-2 Small

To train with the **GPT-2 Small** architecture (124M parameters) on the **Wikitext-103** dataset, simply change the base configuration file in `configs/train.yaml` to point to the corresponding full config.
No other modifications are needed unless you want custom hyperparameters.

### ⏯️ Resuming Training

To resume training from a saved checkpoint, include the path to that checkpoint at the top of your model configuration file, see `configs/full_configs/simple_en_wiki_resume_checkpoint.yaml` for an example.
Once the path is added, running the training command will automatically continue from the latest saved state.

### 💉 Fact Injection Experiments

To run fact-injection experiments, enable the `inject` block inside the trainer settings. This activates the components responsible for:

- **Data Mixing:** Combining your primary training dataset with a stream of injected facts.
- **Evaluation:** Periodically probing the model with targeted prompts to measure how well the injected facts were learned and retained.

You can fully customize the injection process, including frequency, mixing strategy, and data source, by adjusting the corresponding fields in the config file.

A reference configuration is available in `configs/full_configs/insert_data_simple_en_wiki.yaml`.

#### 🎲 Fake Injection Data Generation

The repository includes utilities for generating fake data for injection experiments. You can find these utilities in the `scripts/` directory.

To generate fake data for injection, you can use the `generate_injections.py` script. This script reads a configuration file (e.g., `inject_config.json`) that defines the templates and test cases for the injected facts, and generates the data files needed for training.

**Example Usage:**

```bash
uv run scripts/generate_injections.py --save-path data/insert/test/ --inject-config test/insert_config.json --num_injections 1 --seed 1 --no-shuffle
```

**Arguments:**

- `--save-path`: Directory to save the generated files.
- `--inject-config`: Path to the JSON config file containing the injection templates (relative to `data/insert/`).
- `--num_injections`: Number of injections to generate for each fact template.
- `--seed`: Random seed for reproducibility.
- `--no-shuffle`: Disable shuffling of the generated data.

The configuration file (e.g., `data/insert/test/insert_config.json`) should contain the templates for the facts you want to inject, along with the corresponding test cases (prompts and completions) to evaluate the model's knowledge of these facts.

> **Important:** After generating the data, remember to update your training configuration file (e.g., `configs/full_configs/insert_data_simple_en_wiki.yaml`) to point to the newly generated files in the `insert` section (e.g., `insert_data: test/injected_data.txt`).
