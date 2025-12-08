<div align="center">

<h1>

**Tidy Super Tiny Language Models**

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

## How to Configure

## How to Use

### Traning Simple English Wikipedia Model

### Training GPT-2 Small

### Training with Injection

## Report on Injection Experiments

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