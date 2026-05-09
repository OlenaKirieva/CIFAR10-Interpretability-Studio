"""
Utility Module
Provides helper functions for configuration management, logging, 
and path resolution across different operating systems.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

# Root directory of the dashboard project
BASE_DIR = Path(__file__).resolve().parent.parent

# Dictionary containing metadata for different model architectures
MODEL_DESCRIPTIONS = {
    "SimpleCNN": {
        "layers": 2,
        "params": "~50k",
        "description": "Basic convolutional network. Two Conv2D + MaxPooling blocks. "
        "Designed to detect simple patterns.",
        "features": ["Dropout 0.3"],
    },
    "ProCNN_8Layers_Fast": {
        "layers": 8,
        "params": "~1.2M",
        "description": "Deep VGG-style architecture. Four blocks with two convolutions each. "
        "High capacity for complex visual features.",
        "features": ["Batch Normalization", "Agressive Augmentation"],
    },
    "ProCNN_8Layers_Optimal": {
        "layers": 8,
        "params": "~1.2M",
        "description": "Deep VGG-style architecture. Four blocks with two convolutions each. "
        "Utilizes a Scheduler for fine-tuning weights at later training stages.",
        "features": [
            "Batch Normalization",
            "Learning Rate Scheduler",
            "Early Stopping",
        ],
    },
}


def setup_logging() -> None:
    """
    Initializes the global logging system for the dashboard.
    Configures log level, formatting, and outputs logs to both
    the system console and a local 'dashboard.log' file.
    """
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        # File handler ensures logs persist after the app session ends
        file_handler = logging.FileHandler(
            str(BASE_DIR / "dashboard.log"), encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def load_config(config_path: str = "config/dashboard_config.yaml") -> Dict[str, Any]:
    """
    Loads and parses a YAML configuration file.

    Args:
        config_path (str): The path to the YAML file relative to the project root.

    Returns:
        Dict[str, Any]: A dictionary containing configuration parameters.
    """
    with open(BASE_DIR / config_path, "r") as f:
        return yaml.safe_load(f)


def get_absolute_path(relative_path: str) -> str:
    """
    Resolves a relative file path to an absolute system path.
    Essential for cross-platform compatibility (Windows local vs Streamlit Cloud Linux).

    Args:
        relative_path (str): A path string (e.g., '../data/processed/file.csv').

    Returns:
        str: An absolute path string recognized by the operating system.
    """
    # If the path is already absolute (contains ':' on Windows or starts with / on Linux)
    if ":" in relative_path or relative_path.startswith("/"):
        return relative_path

    # Resolve relative path based on the project's base directory
    return str((BASE_DIR / relative_path).resolve())
