"""Utility functions for checkpoint management."""

import glob
import os
from typing import Any, Dict, List, Optional

import torch

from utils.logger import get_logger

logger = get_logger(__name__)


def find_latest_checkpoint(checkpoint_dir: str, pattern: str = "*.pt") -> Optional[str]:
    """Find the latest checkpoint file in a directory.

    Args:
        checkpoint_dir (str): Directory to search for checkpoints
        pattern (str): File pattern to match (default: "*.pt")

    Returns:
        Optional[str]: Path to the latest checkpoint file, or None if none found
    """
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
        return None

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not checkpoint_files:
        logger.warning(f"No checkpoint files found in {checkpoint_dir}")
        return None

    # Sort by modification time (newest first)
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def find_checkpoint_by_iteration(checkpoint_dir: str, iteration: int) -> Optional[str]:
    """Find a checkpoint file by iteration number.

    Args:
        checkpoint_dir (str): Directory to search for checkpoints
        iteration (int): Iteration number to find

    Returns:
        Optional[str]: Path to the checkpoint file, or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
        return None

    # Look for files with the iteration number in the filename
    pattern = f"*_{iteration}.pt"
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))

    if not checkpoint_files:
        logger.warning(f"No checkpoint found for iteration {iteration} in {checkpoint_dir}")
        return None

    # Return the first match (should be unique)
    checkpoint_path = checkpoint_files[0]
    logger.info(f"Found checkpoint for iteration {iteration}: {checkpoint_path}")
    return checkpoint_path


def list_checkpoints(checkpoint_dir: str, pattern: str = "*.pt") -> List[Dict[str, Any]]:
    """List all checkpoint files in a directory with metadata.

    Args:
        checkpoint_dir (str): Directory to search for checkpoints
        pattern (str): File pattern to match (default: "*.pt")

    Returns:
        List[Dict[str, Any]]: List of dictionaries with checkpoint metadata
    """
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
        return []

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    checkpoints = []

    for checkpoint_path in checkpoint_files:
        try:
            # Get file metadata
            stat = os.stat(checkpoint_path)
            filename = os.path.basename(checkpoint_path)

            # Try to extract iteration from filename
            iteration = None
            if "_" in filename:
                try:
                    iteration = int(filename.split("_")[-1].replace(".pt", ""))
                except ValueError:
                    pass

            # Try to load checkpoint to get more info
            checkpoint_info = {
                "path": checkpoint_path,
                "filename": filename,
                "size_mb": stat.st_size / (1024 * 1024),
                "modified_time": stat.st_mtime,
                "iteration": iteration,
            }

            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if "iteration" in checkpoint:
                    checkpoint_info["iteration"] = checkpoint["iteration"]
                if "config" in checkpoint:
                    checkpoint_info["dataset"] = (
                        checkpoint["config"].get("trainer", {}).get("dataset", "unknown")
                    )
            except Exception as e:
                logger.warning(f"Could not load checkpoint {checkpoint_path}: {e}")
                checkpoint_info["error"] = str(e)

            checkpoints.append(checkpoint_info)

        except Exception as e:
            logger.warning(f"Error processing checkpoint {checkpoint_path}: {e}")

    # Sort by iteration number (if available) or modification time
    checkpoints.sort(key=lambda x: (x.get("iteration", 0), x["modified_time"]), reverse=True)

    return checkpoints


def validate_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Validate a checkpoint file and return information about it.

    Args:
        checkpoint_path (str): Path to the checkpoint file

    Returns:
        Dict[str, Any]: Validation results and checkpoint information
    """
    result = {"valid": False, "path": checkpoint_path, "errors": [], "warnings": [], "info": {}}

    if not os.path.exists(checkpoint_path):
        result["errors"].append(f"Checkpoint file does not exist: {checkpoint_path}")
        return result

    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Check required keys
        required_keys = ["model", "optimizer", "iteration", "config"]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            result["errors"].append(f"Missing required keys: {missing_keys}")
        else:
            result["info"]["iteration"] = checkpoint["iteration"]
            result["info"]["has_lr_scheduler"] = "lr_scheduler" in checkpoint
            result["info"]["has_dropout_scheduler"] = "dropout_scheduler" in checkpoint
            result["info"]["has_scaler"] = "scaler" in checkpoint
            result["info"]["has_random_states"] = "torch_rng_state" in checkpoint

            # Extract config info
            if "config" in checkpoint:
                config = checkpoint["config"]
                result["info"]["dataset"] = config.get("trainer", {}).get("dataset", "unknown")
                result["info"]["model_type"] = config.get("model", {}).get(
                    "model_shell_type", "unknown"
                )

        # Check for optional but recommended keys
        optional_keys = [
            "lr_scheduler",
            "dropout_scheduler",
            "scaler",
            "torch_rng_state",
            "numpy_rng_state",
        ]
        missing_optional = [key for key in optional_keys if key not in checkpoint]
        if missing_optional:
            result["warnings"].append(f"Missing optional keys: {missing_optional}")

        # If no errors, mark as valid
        if not result["errors"]:
            result["valid"] = True

    except Exception as e:
        result["errors"].append(f"Error loading checkpoint: {str(e)}")

    return result


def cleanup_old_checkpoints(
    checkpoint_dir: str, keep_last: int = 5, pattern: str = "*.pt"
) -> List[str]:
    """Clean up old checkpoint files, keeping only the most recent ones.

    Args:
        checkpoint_dir (str): Directory containing checkpoints
        keep_last (int): Number of recent checkpoints to keep
        pattern (str): File pattern to match (default: "*.pt")

    Returns:
        List[str]: List of deleted checkpoint file paths
    """
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
        return []

    checkpoints = list_checkpoints(checkpoint_dir, pattern)

    if len(checkpoints) <= keep_last:
        logger.info(f"Only {len(checkpoints)} checkpoints found, no cleanup needed")
        return []

    # Keep the most recent checkpoints
    checkpoints_to_keep = checkpoints[:keep_last]
    checkpoints_to_delete = checkpoints[keep_last:]

    deleted_files = []
    for checkpoint in checkpoints_to_delete:
        try:
            os.remove(checkpoint["path"])
            deleted_files.append(checkpoint["path"])
            logger.info(f"Deleted old checkpoint: {checkpoint['path']}")
        except Exception as e:
            logger.error(f"Error deleting checkpoint {checkpoint['path']}: {e}")

    logger.info(
        f"Cleaned up {len(deleted_files)} old checkpoints, kept {len(checkpoints_to_keep)} recent ones"
    )
    return deleted_files
