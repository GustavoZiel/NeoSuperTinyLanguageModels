import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to sys.path to allow imports from models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from models.build_models import build_model


def load_model(path):
    """Loads a model from a local path."""
    print(f"Loading model from: {path}...")
    # Load checkpoint with weights_only=False to support custom objects like OmegaConf
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint=checkpoint, verbose=True)
    return model


def calculate_layer_divergence(model_base, model_new):
    """Calculates the Relative L2 Norm difference between two models per parameter group.
    Formula: ||W_new - W_base|| / ||W_base||
    """
    print("Calculating weight divergence...")

    layer_diffs = defaultdict(list)
    global_diffs = {}

    params_base = dict(model_base.named_parameters())
    params_new = dict(model_new.named_parameters())

    # Verify architectures match
    if params_base.keys() != params_new.keys():
        print(
            "WARNING: Model architectures do not match exactly. Calculating intersection only."
        )

    for name, p_base in params_base.items():
        if name not in params_new:
            continue

        p_new = params_new[name]

        # Calculate L2 Norm of the difference
        diff_tensor = p_new - p_base
        l2_diff = torch.norm(diff_tensor).item()
        l2_base = torch.norm(p_base).item()

        # Avoid division by zero
        relative_change = l2_diff / (l2_base + 1e-8)

        # Store for plotting
        global_diffs[name] = relative_change

        # Heuristic to group by Layer Number (works for Llama, GPT, Pythia, etc.)
        # Looks for patterns like "layers.0.", "h.1.", "blocks.12."
        layer_match = re.search(r"\.(\d+)\.", name)

        if layer_match:
            layer_idx = int(layer_match.group(1))
            layer_diffs[layer_idx].append(relative_change)
        elif "embed" in name or "wte" in name:
            layer_diffs["Embedding"].append(relative_change)
        elif (
            "head" in name or "output" in name or "norm" in name and "layer" not in name
        ):
            layer_diffs["Output/Head"].append(relative_change)

    # Average the changes per layer group
    averaged_layer_diffs = {}
    for key, vals in layer_diffs.items():
        averaged_layer_diffs[key] = np.mean(vals)

    return averaged_layer_diffs, global_diffs


def plot_divergence(averaged_diffs):
    """Plots the layer-wise changes."""
    # Sort keys: Embed -> 0, 1, 2... -> Output
    sorted_keys = []
    if "Embedding" in averaged_diffs:
        sorted_keys.append("Embedding")

    # Extract integer layers and sort them
    int_layers = sorted([k for k in averaged_diffs.keys() if isinstance(k, int)])
    sorted_keys.extend(int_layers)

    if "Output/Head" in averaged_diffs:
        sorted_keys.append("Output/Head")

    values = [averaged_diffs[k] for k in sorted_keys]
    labels = [str(k) for k in sorted_keys]

    plt.figure(figsize=(12, 6))

    # Create gradient bars based on intensity
    bars = plt.bar(labels, values, color=plt.cm.viridis(np.array(values) / max(values)))

    plt.title("Weight Deterioration Analysis: Where did the model change?", fontsize=16)
    plt.xlabel("Model Layers (Embed -> Layers -> Head)", fontsize=12)
    plt.ylabel("Relative Weight Change (L2 Norm)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate max change
    max_val = max(values)
    max_idx = values.index(max_val)
    plt.text(
        max_idx, max_val, f"{max_val:.4f}", ha="center", va="bottom", fontweight="bold"
    )

    plt.tight_layout()
    output_file = "divergence_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    # plt.show()


# ==========================================
# EXECUTION BLOCK
# ==========================================

# Replace these paths with your actual model folders
path_baseline = "/home/ziel/codes/SIPGA/NeoSuperTinyLanguageModels/vast_checkpoints/baseline-v6-simple_en_wiki/20251112_1324_simple_en_wiki_40_epochs.pt"
path_retrained = "/home/ziel/codes/SIPGA/NeoSuperTinyLanguageModels/vast_checkpoints/baseline-v6-simple_en_wiki/20251112_1424_simple_en_wiki_50_epochs.pt"
# path_retrained = "/home/ziel/codes/SIPGA/NeoSuperTinyLanguageModels/vast_checkpoints/baseline-v6-paragraphs-simple_en_wiki_1_epoch_16_inserted/20251112_1427_simple_en_wiki_50_epochs_insert_uniform_16insertions.pt"
# path_retrained = "/home/ziel/codes/SIPGA/NeoSuperTinyLanguageModels/vast_checkpoints/baseline-v6-simple_en_wiki_1_epoch_1_inserted/20251112_1423_simple_en_wiki_50_epochs_insert_uniform_1insertions.pt"

# UNCOMMENT THE LINES BELOW TO RUN
model_a = load_model(path_baseline)
model_b = load_model(path_retrained)

avg_diffs, full_diffs = calculate_layer_divergence(model_a, model_b)
plot_divergence(avg_diffs)
