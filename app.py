"""
CIFAR-10 Interpretability Studio
Main entry point for the Streamlit dashboard.
Organized into a modular architecture for professional deployment.
"""

import logging
import os
import sys
from typing import Optional

import mlflow  # type: ignore
import pandas as pd
import streamlit as st

# Ensure internal modules are discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.data_loader import prepare_test_data_locally
from src.mlflow_utils import (
    get_all_experiments,
    init_mlflow,
    load_artifact_text,
    load_model_smart,
    load_predictions_from_mlflow,
)
from src.tabs.dataset_tab import render_dataset_tab
from src.tabs.error_tab import render_error_tab
from src.tabs.explain_tab import render_explain_tab
from src.utils import MODEL_DESCRIPTIONS, load_config, setup_logging

# Professional page configuration
st.set_page_config(
    page_title="CIFAR-10 Interpretability Studio", page_icon="🧠", layout="wide"
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Main dashboard orchestration logic."""
    setup_logging()
    config = load_config()

    # --- 1. DATA PREPARATION ---
    # Downloads and prepares 10,000 images on the first run
    df_full = prepare_test_data_locally()

    # --- 2. SIDEBAR: MODEL SELECTION ---
    st.sidebar.header("📦 Model Management")

    # RELIABLE CONNECTION: Using a relative path for SQLite compatibility
    db_file = config["mlflow"]["db_path"]
    mlflow_uri = f"sqlite:///{db_file}"
    init_mlflow(mlflow_uri)
    logger.info(f"Connected to MLflow Database: {mlflow_uri}")

    try:
        all_exps = get_all_experiments()
        # Locate the research experiment folder
        selected_exp = next(
            (e for e in all_exps if e.name == "CIFAR10_Final_Research"), None
        )

        if selected_exp is None:
            if all_exps:
                selected_exp = all_exps[0]
                st.sidebar.warning(
                    f"Research exp not found. Using: {selected_exp.name}"
                )
            else:
                st.sidebar.error("No MLflow database records found.")
                st.stop()

        # Fetch runs and filter for successful iterations
        runs_df = mlflow.search_runs(experiment_ids=[selected_exp.experiment_id])
        if runs_df.empty:
            st.sidebar.error("No valid runs found in the database.")
            st.stop()

        finished_runs = runs_df[runs_df["status"] == "FINISHED"].copy()

        # Architecture filtering: Best SimpleCNN + All ProCNN versions
        is_simple = finished_runs["tags.mlflow.runName"].str.contains(
            "Simple", na=False
        )
        best_simple = (
            finished_runs[is_simple]
            .sort_values(by="metrics.val_acc", ascending=False)
            .head(1)
        )
        all_pro = finished_runs[~is_simple].sort_values(
            by="metrics.val_acc", ascending=False
        )
        final_runs = pd.concat([all_pro, best_simple])

        selected_run_name = st.sidebar.selectbox(
            "Select Model Version",
            final_runs["tags.mlflow.runName"].fillna("Unnamed").tolist(),
        )

        run_info = final_runs[
            final_runs["tags.mlflow.runName"] == selected_run_name
        ].iloc[0]
        run_id = run_info["run_id"]

        # 3. RESOURCE LOADING
        # Load weights and pre-calculated predictions
        model = load_model_smart(
            run_id, selected_exp.experiment_id, selected_run_name, mlflow_uri
        )
        df_preds_raw = load_predictions_from_mlflow(
            run_id, mlflow_uri, selected_exp.experiment_id
        )

        # Synchronize MLflow results with local storage paths
        df_preds: Optional[pd.DataFrame] = None
        if df_preds_raw is not None:
            min_len = min(len(df_preds_raw), len(df_full))
            df_preds = df_preds_raw.iloc[:min_len].copy()
            df_preds["image_path"] = df_full["image_path"].iloc[:min_len].values

            if model:
                st.sidebar.success(f"✅ Model Loaded\n{selected_run_name[:30]}...")

        # 4. MODEL PASSPORT (Metadata Display)
        st.sidebar.divider()
        st.sidebar.subheader("📄 Model Passport")

        if "Simple" in selected_run_name:
            m_key = "SimpleCNN"
        elif "E:100" in selected_run_name:
            m_key = "ProCNN_8Layers_Optimal"
        else:
            m_key = "ProCNN_8Layers_Fast"

        info = MODEL_DESCRIPTIONS.get(m_key, {})
        st.sidebar.write(f"**Arch:** {info.get('description', 'N/A')}")
        st.sidebar.write(
            f"**Complexity:** {info.get('layers')} Layers | {info.get('params')} Params"
        )

    except Exception as e:
        st.sidebar.error(f"Setup Error: {e}")
        st.stop()

    # --- 5. MAIN DASHBOARD TABS ---
    t1, t2, t3 = st.tabs(
        ["📊 Dataset Exploration", "🔍 Error Analysis", "🧠 Model Interpretation"]
    )

    with t1:
        render_dataset_tab(config, df_full)

    with t2:
        # Pass the model object for error inspection if needed
        render_error_tab(
            config, df_preds, run_id, selected_exp.experiment_id, mlflow_uri
        )

    with t3:
        render_explain_tab(config, df_full, model, selected_run_name, info)


if __name__ == "__main__":
    main()
