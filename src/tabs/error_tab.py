"""
Error Analysis Tab Module
Provides a comprehensive interface for auditing model misclassifications.
Includes global metric visualization and an interactive instance-level inspector.
"""

import pandas as pd
import streamlit as st

from src.analytics import get_filtered_errors
from src.data_loader import load_image
from src.mlflow_utils import load_artifact_text
from src.ui_components import render_classification_report_mini, render_error_matrix


def render_error_tab(
    config: dict, df_preds: pd.DataFrame,  run_id: str, exp_id: str, mlflow_uri: str
) -> None:
    """
    Renders the Error Explorer tab layout.

    Args:
        config (dict): Global configuration dictionary.
        df_preds (pd.DataFrame): DataFrame containing model predictions for the test set.
        run_id (str): MLflow Run ID.
        exp_id (str): MLflow Experiment ID.
        mlflow_uri (str): URI for the MLflow tracking server.
    """
    if df_preds is None:
        st.warning("Prediction artifacts (CSV) not found for this model version.")
        return

    # Analytical Overview Row (Matrix & Report)
    col_matrix, col_report = st.columns([1, 1])

    with col_matrix:
        render_error_matrix(df_preds, config["classes"])

    with col_report:
        report_txt = load_artifact_text(
            run_id, exp_id, mlflow_uri, "classification_report.txt"
        )
        render_classification_report_mini(report_txt)

    st.markdown("---")

    # Interactive Instance Inspector

    # Session state management for UI stability
    if "last_run_id" not in st.session_state or st.session_state.last_run_id != run_id:
        st.session_state.last_run_id = run_id
        st.session_state.t2_idx = 0

    col_filter, col_preview = st.columns([1, 1])

    with col_filter:
        st.write("##### 🛠️ Filters")
        f_col1, _, f_col2 = st.columns([1, 0.2, 1])

        with f_col1:
            # Class filtering options
            t_f_name = st.selectbox(
                "Select Actual Class:",
                ["All"] + config["classes"],
                key="t2_filter_true",
            )

            p_f_name = st.selectbox(
                "Model Predicted as:",
                ["All Errors", "Correct Only"] + config["classes"],
                key="t2_filter_pred",
            )

        with f_col2:
            # Sorting logic
            sort_mode = st.radio(
                "Sort results by:",
                ["Highest Confidence", "Lowest Confidence"],
                key="t2_sort",
            )

        # Apply filtering and sorting logic from analytics module
        display_df = get_filtered_errors(
            df_preds, t_f_name, p_f_name, config["classes"], sort_mode
        )

        if not display_df.empty:
            st.write(f"🔍 Found **{len(display_df)}** matching samples.")
            idx_in_list = st.slider(
                "Browse matches", 0, len(display_df) - 1, key="t2_slider"
            )
            selected_row = display_df.iloc[idx_in_list]
            global_idx = display_df.index[idx_in_list]
        else:
            st.warning("No samples found for this combination.")

        st.caption("💡 Tip: Each class in CIFAR-10 test set contains 1000 samples.")

    with col_preview:
        st.write("##### 🖼️ Sample Preview")
        if not display_df.empty:
            # Nested columns for compact metadata display
            img_col, info_col = st.columns([1, 1])

            with img_col:
                image = load_image(selected_row["image_path"])
                if image:
                    st.image(image, width=240)
                st.caption(f"**Global Test Index:** `{global_idx}`")

            with info_col:
                st.info(
                    f"**True Value:** {config['classes'][selected_row['true_label']]}"
                )

                # Success/Error color coding
                if selected_row["true_label"] == selected_row["pred_label"]:
                    st.success(
                        f"**Prediction:** {config['classes'][selected_row['pred_label']]} (Correct)"
                    )
                else:
                    st.error(
                        f"**Prediction:** {config['classes'][selected_row['pred_label']]} (Incorrect)"
                    )

                conf_score = selected_row["confidence"]
                st.write(f"**Confidence:** {conf_score:.2%}")

            st.caption(
                "💡 Tip: Copy the Global Index to use in the Explainability tab."
            )
