"""
Dataset Exploration Tab Module
Provides UI components for high-level statistics and interactive 
data inspection of the CIFAR-10 test set.
"""

import streamlit as st

from src.data_loader import load_image
from src.ui_components import render_global_stats


def render_dataset_tab(config: dict, df_full) -> None:
    """
    Renders the dataset exploration tab layout.

    Args:
        config (dict): Configuration dictionary containing class names.
        df_full (pd.DataFrame): The complete test dataset registry.
    """
    # Maintain the preferred column ratio for a balanced UI layout
    col_stats, col_inspector = st.columns([1.5, 1.2])

    with col_stats:
        render_global_stats(config["classes"])

    with col_inspector:
        st.subheader("Interactive Inspector")

        if df_full is not None:
            # 1. Category Filtering
            selected_class = st.selectbox(
                "Category", ["All"] + config["classes"], key="t1_cls"
            )

            filtered_df = (
                df_full
                if selected_class == "All"
                else df_full[
                    df_full["label"] == config["classes"].index(selected_class)
                ]
            )

            # 2. Sample Selection (Dynamic range based on filter)
            max_index = len(filtered_df) - 1
            sample_idx = st.number_input(
                f"Sample Index (0 - {max_index})", 0, max_index, 0
            )

            # 3. Visualization
            selected_sample = filtered_df.iloc[sample_idx]
            image = load_image(selected_sample["image_path"])

            if image:
                # Keep images compact at 200px as per the original design
                st.image(image, width=200)

                # Display classification info in a standard info box
                st.info(
                    f"**True Category:** {config['classes'][selected_sample['label']]}"
                )

                # Metadata for cross-referencing with other tabs
                st.caption(f"Global Registry Index: {filtered_df.index[sample_idx]}")
        else:
            st.warning("Dataset registry not initialized.")
