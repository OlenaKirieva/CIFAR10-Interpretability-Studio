"""
CIFAR-10 Interpretability Studio
Main entry point for the Streamlit dashboard.
Handles model selection, data synchronization, and tab navigation.
"""

import os
import sys
from typing import Optional

import mlflow  # type: ignore
import pandas as pd
import streamlit as st

# Ensure the 'src' directory is in the python path for module discovery
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.data_loader import prepare_test_data_locally
from src.mlflow_utils import (
    get_all_experiments,
    init_mlflow,
    load_model_smart,
    load_predictions_from_mlflow,
)
from src.tabs.dataset_tab import render_dataset_tab
from src.tabs.error_tab import render_error_tab
from src.tabs.explain_tab import render_explain_tab
from src.utils import MODEL_DESCRIPTIONS, get_absolute_path, load_config, setup_logging

# Page configuration for a wide-screen professional look
st.set_page_config(
    page_title="CIFAR-10 Interpretability Studio", page_icon="🧠", layout="wide"
)


def main() -> None:
    """Main dashboard logic."""
    setup_logging()

    try:
        config = load_config()
    except Exception as e:
        st.error(f"Critical: Configuration file not found. {e}")
        st.stop()

    # --- 1. DATA INITIALIZATION ---
    # Automatically prepares 10,000 test images locally if not present
    df_full = prepare_test_data_locally()

    # --- 2. SIDEBAR: MODEL MANAGEMENT ---
    st.sidebar.header("📦 Model Selection")

    # Dynamic database path resolution (Avoids hard-coded absolute paths)
    db_path = get_absolute_path(config["mlflow"]["db_path"])
    mlflow_uri = f"sqlite:///{db_path.replace('\\', '/')}"
    init_mlflow(mlflow_uri)

    try:
        all_exps = get_all_experiments()
        # Filter for the research experiment defined in Lab 4.1
        selected_exp = next(
            (e for e in all_exps if e.name == "CIFAR10_Final_Research"), None
        )

        if selected_exp is None:
            st.sidebar.warning(
                "Target experiment not found in MLflow. Showing all available."
            )
            if all_exps:
                selected_exp = all_exps[0]
            else:
                st.sidebar.error("No experiments found. Ensure mlflow.db is present.")
                st.stop()

        # Retrieve runs and filter for successful completions
        runs_df = mlflow.search_runs(experiment_ids=[selected_exp.experiment_id])
        if runs_df.empty:
            st.sidebar.error("No runs found in the selected experiment.")
            st.stop()

        finished_runs = runs_df[runs_df["status"] == "FINISHED"].copy()

        # Model Version Filtering Logic: Keep 1 best Baseline + All Pro versions
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

        # Interactive Model Selection
        selected_run_name = st.sidebar.selectbox(
            "Select Architecture Version",
            final_runs["tags.mlflow.runName"].fillna("Unnamed").tolist(),
        )
        run_info = final_runs[
            final_runs["tags.mlflow.runName"] == selected_run_name
        ].iloc[0]
        run_id = run_info["run_id"]

        # 3. RESOURCE LOADING
        # Smart load model weights and pre-calculated predictions
        model = load_model_smart(
            run_id, selected_exp.experiment_id, selected_run_name, mlflow_uri
        )
        df_preds_raw = load_predictions_from_mlflow(
            run_id, mlflow_uri, selected_exp.experiment_id
        )

        # Synchronize MLflow predictions with local image paths
        df_preds: Optional[pd.DataFrame] = None
        if df_preds_raw is not None:
            min_len = min(len(df_preds_raw), len(df_full))
            df_preds = df_preds_raw.iloc[:min_len].copy()
            df_preds["image_path"] = df_full["image_path"].iloc[:min_len].values

            if model:
                st.sidebar.success(f"✅ {selected_run_name} \n Ready")

        # 4. MODEL PASSPORT (Sidebar UI)
        st.sidebar.divider()
        st.sidebar.subheader("📄 Model Passport")

        # Determine metadata key
        if "Simple" in selected_run_name:
            m_key = "SimpleCNN"
        elif "E:100" in selected_run_name:
            m_key = "ProCNN_8Layers_Optimal"
        else:
            m_key = "ProCNN_8Layers_Fast"

        info = MODEL_DESCRIPTIONS.get(m_key, {})
        st.sidebar.write(f"**Description:** {info.get('description', 'N/A')}")
        st.sidebar.write(
            f"**Complexity:** {info.get('layers')} Layers | {info.get('params')} Params"
        )

        # with st.sidebar.expander("📈 View Training Artifact (TXT)"):
        #     report = load_artifact_text(run_id, selected_exp.experiment_id, mlflow_uri, "classification_report.txt")
        #     st.code(report if report else "Report not available for this run.", language="text")

    except Exception as e:
        st.sidebar.error(f"Initialization Failure: {e}")
        st.stop()

    # --- 5. MAIN NAVIGATION (Tabs) ---
    t1, t2, t3 = st.tabs(
        ["📊 Dataset Exploration", "🔍 Error Analysis", "🧠 Model Interpretation"]
    )

    with t1:
        render_dataset_tab(config, df_full)

    with t2:
        render_error_tab(
            config, df_preds, run_id, selected_exp.experiment_id, mlflow_uri
        )

    with t3:
        render_explain_tab(config, df_full, model, selected_run_name, info)


if __name__ == "__main__":
    main()
