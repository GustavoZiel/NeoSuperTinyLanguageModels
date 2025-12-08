<div align="center">

<h1>

<!-- **Tidy Super Tiny Language Models** -->
**Tidy STLMs**

<h3>

<!-- [Documentation](docs/) • [Experiments](docs/experiment.md) • [Report](https://wandb.ai/gustavogrib-ggr-usp/adaptive-pdf-extractor/reports/Adaptative-PDF-Extractor-Analysis--VmlldzoxNDk4MjY0OQ?accessToken=sdl3m4ghmnv8tdnho85ia68qoxi88phpr9xp0pduj0lnjwfwwju1lg9fn38rr5tw) -->

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)

</div>

This repository is a tidier version developed by me during my stay on the SIPGA (Singapore International Pre-Graduate Award) program. It's based on the original [Super Tiny Language Models Repo](), follow the link to access the original repository and its original structure, related papers and dedicated citation information.

> **Disclaimer:** For a comprehensive understanding of the original Super Tiny Language Models, including its foundational concepts, broader applications, and published paper, please refer to the original repository.

The main goal of this version is to leverage the existing codebase to focus on my research topic: How new, never-seen before small factoids can be injected into the knowledge base of Language Models, and how the model behaves when prompted with these injected facts.

Therefore, this version adds new scripts, utilities, and documentation specifically tailored for injection-based research.

Main differences in this version include:
- Focus on injection-based research
- Removal of unnecessary files and directories not relevant to the injection research or not used in the experiments
- Improved documentation and instructions for running experiments
  Broader configurability for training and evaluation, with runtime generation of prompts
- Enhanced logging and experiment tracking using Weights & Biases (wandb)
- Scripts and utilities specifically tailored for injection research

> **Note:** It's possible to disable injection-specific features and run standard training and evaluation by adjusting the configuration files accordingly. So this repo can still be used for general STLM training and evaluation.

## How to Install

> Requires: Python 3.11+ and [uv](https://uv.run/) installed.

```bash
# Clone the repository
git clone <repo_url>
cd tidy-stlms

# Synchronize and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Ready to go!
```

## How to Configure

> Configuration files are located in the `configs/` directory. You can modify these YAML files to set hyperparameters, dataset paths, injection settings, and other options.

Every configuration file follows the same structure, and you can create new configuration files by following the existing ones.

The main parts are:

- `defaults`: Specifies the base configuration files to use. You can inherit from multiple base configurations.
- `model`: Defines the model architecture and parameters.
- `dataset`: Specifies the dataset to use for training and evaluation.
- `training`: Contains training hyperparameters such as learning rate, batch size, number of epochs, etc.
- `injection`: (Optional) Settings related to injection experiments, such as the type of injection, data generation methods, and evaluation metrics.

## How to Use

### Traning Simple English Wikipedia Model

This is the default configuration and can be run as soon as the installation steps are completed.

```bash
uv run src/train.py
```

It's just a taste, the default file will run for only 10 steps and should not take more than 5 minutes. In the console, you should see all the logging messages in the, and as soon as the training starts, you should see the progress bar and other metrics. Most important, see the loss decreasing.

From there, you can increase the number of training steps in the `configs/train.yaml` file, epochs, batch size, learning rate, model architecture, etc.

### Training GPT-2 Small

Each file in the `configs/full_configs/` directory represents a full configuration for training a specific model on a specific dataset. For example, to train a GPT-2 Small model on the Simple English Wikipedia dataset, you can run:

```bash
uv run src/train.py --config configs/full_configs/gpt2_simple_en_wiki.yaml
```

You can create as many configuration files as you want, just make sure to follow the structure of the existing files, and change the path of the .yaml file to use for training on the `train.yaml` file.

### Training with Injection

To run training with injection experiments, you can use the configuration files that include injection settings. For example:

```bash
uv run src/train.py --config configs/full_configs/insert_data_simple_en_wiki.yaml
```

The ideia is to see how the model behaves when new facts are injected into its training data. You can customize the injection settings in the configuration file to explore different scenarios. How much data to inject, with what frequency, what type of data, etc.

#### Fake Injection Data Generation

The repository includes utilities for generating fake data for injection experiments. You can find these utilities in the `src/injection/` directory. You can customize the data generation process by modifying the relevant scripts or creating new ones.

## Report on Injection Experiments

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
  - Link Wandb Report