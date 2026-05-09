"""
Explainability Tab Module
Provides tools for model interpretability using Grad-CAM and LIME.
Supports categorical browsing, global indexing, and custom uploads.
"""

import pandas as pd
import streamlit as st
from PIL import Image

from src.data_loader import load_image
from src.explainability import run_lime
from src.inference import predict_image, run_gradcam
from src.ui_components import (
    create_report_image,
    render_prediction_box,
    render_probability_chart,
)


def render_explain_tab(
    config: dict, df_full: pd.DataFrame, model, run_name: str, model_info: dict
) -> None:
    """
    Renders the Model Interpretation Studio.

    Args:
        config: Configuration dictionary.
        df_full: Full test dataset registry.
        model: Loaded PyTorch model.
        run_name: Name of the current MLflow run.
        model_info: Metadata dictionary for the current architecture.
    """

    # 1. TOP SELECTION ROW
    c_mode, c_method = st.columns(2)
    with c_mode:
        src_mode = st.radio(
            "Data Source",
            ["Browse by Category", "Global Index", "Upload File"],
            horizontal=True,
        )
    with c_method:
        method = st.radio("Explanation Method", ["Grad-CAM", "LIME"], horizontal=True)

    img, true_label, current_id = None, None, "upload"

    # 2. IMAGE SELECTION LOGIC
    # Dedicated column for selection widgets to save vertical space
    sel_col, _ = st.columns([1, 2])

    with sel_col:
        if src_mode == "Browse by Category":
            selected_cat = st.selectbox(
                "Select Category to explore", config["classes"], key="t3_cat"
            )
            cat_df = df_full[df_full["label"] == config["classes"].index(selected_cat)]
            sub_idx = st.slider(f"Samples in {selected_cat}", 0, len(cat_df) - 1, 0)
            row = cat_df.iloc[sub_idx]
            img = load_image(row["image_path"])
            true_label = selected_cat
            current_id = str(cat_df.index[sub_idx])

        elif src_mode == "Global Index":
            global_idx = st.number_input("Enter Global Index (0-9999)", 0, 9999, 0)
            row = df_full.iloc[global_idx]
            img = load_image(row["image_path"])
            true_label = config["classes"][row["label"]]
            current_id = str(global_idx)

        elif src_mode == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["jpg", "png", "jpeg"],
                label_visibility="collapsed",
            )
            if uploaded_file:
                img = Image.open(uploaded_file).convert("RGB")

    # 3. ANALYSIS AND VISUALIZATION
    if img and model:
        # Perform inference
        pred_idx, conf, tensor, probs_vec = predict_image(model, img)

        # Prepare probability data for charts and reports
        prob_df = pd.DataFrame(
            {"Class": config["classes"], "Prob": probs_vec.cpu().numpy()}
        ).sort_values(by="Prob")

        # Layout: Input | Probability Chart | XAI Result
        res_c1, res_c2, res_c3 = st.columns([1, 1.2, 1])

        with res_c1:
            st.subheader("Target Sample")
            # Display at matched size for better UI consistency
            st.image(img.resize((224, 224)), width=250)
            render_prediction_box(config, pred_idx, conf, true_label)

        with res_c2:
            st.subheader("Confidence")
            render_probability_chart(config["classes"], probs_vec)

        with res_c3:
            st.subheader("Interpretation")
            explanation_data = None
            with st.spinner(f"Running {method}..."):
                if method == "Grad-CAM":
                    explanation_data = run_gradcam(model, tensor, (224, 224))
                else:
                    explanation_data = run_lime(model, img)

                if explanation_data is not None:
                    # Convert to PIL for Streamlit Cloud compatibility
                    st.image(
                        Image.fromarray(explanation_data),
                        width=250,
                        caption=f"{method} View",
                    )
                else:
                    st.error("Failed to generate explanation.")

        # 4. EXPORT FUNCTIONALITY
        if explanation_data is not None:
            st.markdown("---")
            btn_col, _ = st.columns([1, 3])
            with btn_col:
                report_bytes = create_report_image(
                    img.resize((224, 224)),
                    explanation_data,
                    config["classes"][pred_idx],
                    conf,
                    run_name,
                    model_info,
                    method,
                    prob_df,
                )
                st.download_button(
                    label="📥 Download Analytical Report (PNG)",
                    data=report_bytes,
                    file_name=f"report_idx_{current_id}_{method}.png",
                    mime="image/png",
                )
