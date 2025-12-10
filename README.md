<div align="center">

<h1>

**Tidy STLMs - An Enhanced Fork for Fact Injection Research**

<h3>

<!-- [Documentation](docs/) • [Experiments](docs/experiment.md) • [Report](https://wandb.ai/gustavogrib-ggr-usp/adaptive-pdf-extractor/reports/Adaptative-PDF-Extractor-Analysis--VmlldzoxNDk4MjY0OQ?accessToken=sdl3m4ghmnv8tdnho85ia68qoxi88phpr9xp0pduj0lnjwfwwju1lg9fn38rr5tw) -->

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

<!-- ### 🤖 Training GPT-2 Small

Each file in the `configs/full_configs/` directory represents a full configuration for training a specific model on a specific dataset. For example, to train a GPT-2 Small model on the Simple English Wikipedia dataset, you can run:

```bash
uv run src/train.py --config configs/full_configs/gpt2_simple_en_wiki.yaml
```

You can create as many configuration files as you want, just make sure to follow the structure of the existing files, and change the path of the .yaml file to use for training on the `train.yaml` file.

### 💉 Training with Injection

To run training with injection experiments, you can use the configuration files that include injection settings. For example:

```bash
uv run src/train.py --config configs/full_configs/insert_data_simple_en_wiki.yaml
```

The ideia is to see how the model behaves when new facts are injected into its training data. You can customize the injection settings in the configuration file to explore different scenarios. How much data to inject, with what frequency, what type of data, etc.

#### 🎲 Fake Injection Data Generation

The repository includes utilities for generating fake data for injection experiments. You can find these utilities in the `src/injection/` directory. You can customize the data generation process by modifying the relevant scripts or creating new ones.

## 📊 Report on Injection Experiments

A detailed report on the injection experiments, including methodology, results, and analysis, can be found in the [Wandb Report](https://wandb.ai/gustavogrib-ggr-usp/adaptive-pdf-extractor/reports/Adaptative-PDF-Extractor-Analysis--VmlldzoxNDk4MjY0OQ?accessToken=sdl3m4ghmnv8tdnho85ia68qoxi88phpr9xp0pduj0lnjwfwwju1lg9fn38rr5tw). This report provides insights into how the model's knowledge base is affected by the injected data and discusses the implications for future research.

## Overview

- present the repo
  - mention SIPGA program
  - injection focused
  - link to original
  - main differences

- how to run
  - build gpt2
  - build simple_en_wiki with 50M
  - logging and wandb
  
- injection research
  - Questions (maybe, those models are just dump)
  - methodology
    - fake data generator
    - memorization, semantic, syntactic
  - Link fake datasets
  - Link Wandb Report -->
