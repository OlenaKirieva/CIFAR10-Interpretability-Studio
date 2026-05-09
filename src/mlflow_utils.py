import logging
import os
from pathlib import Path
from typing import Any, List, Optional, cast

import mlflow  # type: ignore
import pandas as pd
import streamlit as st
import torch

from src.model import SimpleCNN
from src.model_new import CIFAR10ProCNN

logger = logging.getLogger(__name__)


def _get_base_research_path(tracking_uri: str) -> Path:
    """
    Internal helper to convert MLflow URIs to local filesystem Paths.
    Handles Windows-specific prefixes and SQLite URI formats.
    """
    clean_uri = tracking_uri.replace("sqlite:///", "").replace("file:///", "")
    # Remove leading slash on Windows if it exists (e.g., /C:/... -> C:/...)
    if os.name == "nt" and clean_uri.startswith("/") and ":" in clean_uri:
        clean_uri = clean_uri[1:]

    # Return the parent directory of the DB or mlruns folder
    return Path(clean_uri).parent


def init_mlflow(tracking_uri: str) -> None:
    """Initializes connection to the MLflow tracking server."""
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow connection established: {tracking_uri}")


def get_all_experiments() -> List[Any]:
    """Fetches all available experiments from the tracking server."""
    try:
        return mlflow.search_experiments()
    except Exception as e:
        logger.error(f"Failed to retrieve experiments: {e}")
        return []


@st.cache_resource(show_spinner="Loading model weights...")
def load_model_smart(
    run_id: str, experiment_id: str, run_name: str, tracking_uri: str
) -> Optional[torch.nn.Module]:
    """
    Advanced model loader that scans multiple locations for weights.
    Supports standard MLflow structures and the centralized 'models' folder.
    """
    logger.info(f"Searching for model weights. Run: {run_name} ({run_id})")

    try:
        base_dir = _get_base_research_path(tracking_uri)
        models_dir = base_dir / "mlruns" / "models"
        weights_path = None

        # Phase 1: Search in the centralized 'models' directory (Colab style)
        if models_dir.exists():
            for model_folder in models_dir.iterdir():
                potential_path = model_folder / "artifacts" / "data" / "model.pth"
                meta_path = model_folder / "artifacts" / "MLmodel"

                if potential_path.exists() and meta_path.exists():
                    with open(meta_path, "r") as f:
                        if run_id in f.read():
                            weights_path = potential_path
                            logger.info(
                                f"✅ Found weights in models store: {weights_path}"
                            )
                            break

        # Phase 2: Fallback to standard MLflow run-specific artifacts
        if not weights_path:
            fallback_paths = [
                base_dir
                / "mlruns"
                / str(experiment_id)
                / str(run_id)
                / "artifacts"
                / "model"
                / "data"
                / "model.pth",
                base_dir
                / "mlruns"
                / str(experiment_id)
                / str(run_id)
                / "artifacts"
                / "model.pth",
            ]
            for p in fallback_paths:
                if p.exists():
                    weights_path = p
                    logger.info(f"✅ Found weights in run artifacts: {p}")
                    break

        if not weights_path:
            logger.error(f"❌ Weights file not found for run {run_id}")
            return None

        # Instantiate the correct architecture
        model = (
            CIFAR10ProCNN(n_classes=10)
            if "Pro" in str(run_name)
            else SimpleCNN(n_classes=10)
        )

        # Secure loading with safe globals
        torch.serialization.add_safe_globals([CIFAR10ProCNN, SimpleCNN])
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, torch.nn.Module):
            model = cast(Any, checkpoint)
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    except Exception as e:
        logger.error(f"Critical error during model loading: {e}")
        return None


def load_predictions_from_mlflow(
    run_id: str, tracking_uri: str, experiment_id: str
) -> Optional[pd.DataFrame]:
    """Retrieves pre-calculated prediction CSV files from MLflow artifacts."""
    try:
        base_dir = _get_base_research_path(tracking_uri)

        # Possible locations for prediction artifacts
        search_dirs = [
            base_dir
            / "mlruns"
            / str(experiment_id)
            / str(run_id)
            / "artifacts"
            / "predictions",
            base_dir / "mlruns" / str(experiment_id) / str(run_id) / "artifacts",
        ]

        for pred_dir in search_dirs:
            if pred_dir.exists():
                csv_files = list(pred_dir.glob("*.csv"))
                if csv_files:
                    logger.info(f"Predictions CSV found: {csv_files[0].name}")
                    return pd.read_csv(csv_files[0])
        return None
    except Exception as e:
        logger.error(f"Failed to load predictions: {e}")
        return None


def load_artifact_text(
    run_id: str, experiment_id: str, tracking_uri: str, filename: str
) -> Optional[str]:
    """Reads a textual artifact (like classification reports) from MLflow."""
    try:
        base_dir = _get_base_research_path(tracking_uri)
        file_path = (
            base_dir
            / "mlruns"
            / str(experiment_id)
            / str(run_id)
            / "artifacts"
            / filename
        )

        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return None
    except Exception as e:
        logger.error(f"Failed to load text artifact {filename}: {e}")
        return None
