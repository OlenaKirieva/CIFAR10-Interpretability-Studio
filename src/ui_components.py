"""
UI Components Module
Provides functions for rendering charts, tables, and complex visualizations
within the Streamlit dashboard.
"""

import io
import textwrap
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px  # type: ignore
import plotly.figure_factory as ff  # type: ignore
import streamlit as st
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix  # type: ignore


def render_global_stats(classes: List[str]) -> None:
    """
    Renders high-level dataset statistics and a bar chart of class distribution.

    Args:
        classes: List of CIFAR-10 class names.
    """
    st.subheader("Global Statistics (Full CIFAR-10)")

    # Overview Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", "60,000", help="Full dataset size")
    c2.metric("Training Set", "50,000", "83.3%")
    c3.metric("Test Set", "10,000", "16.7%")

    # Class Distribution Chart
    chart_col, _ = st.columns([0.9, 0.1])
    with chart_col:
        dist_df = pd.DataFrame({"Class": classes, "Count": [6000] * 10})
        fig = px.bar(
            dist_df,
            x="Class",
            y="Count",
            color="Class",
            height=350,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(
            showlegend=False, margin=dict(l=0, r=0, t=20, b=0), xaxis_title=None
        )
        fig.update_traces(width=0.6)
        st.plotly_chart(fig, width="stretch")


def render_classification_report_mini(report_text: Optional[str]) -> None:
    """
    Displays the MLflow classification report in a clean code block.
    """
    st.markdown("##### 📈 Detailed Metrics")
    if report_text:
        st.code(report_text, language="text")
    else:
        st.warning("Metrics report not available for this run.")


def render_error_matrix(df_preds: pd.DataFrame, classes: List[str]) -> None:
    """
    Calculates and renders an interactive Confusion Matrix for the full test set.
    """
    y_true = df_preds["true_label"]
    y_pred = df_preds["pred_label"]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    fig = ff.create_annotated_heatmap(
        z=cm, x=classes, y=classes, colorscale="Viridis", showscale=True
    )

    fig.update_xaxes(side="bottom")
    fig.update_layout(
        title="Confusion Matrix (Full Test Set Analysis)",
        title_x=0.1,
        height=500,
        margin=dict(t=50, b=50),
    )
    st.plotly_chart(fig, width="stretch")


def render_probability_chart(classes: List[str], probs: torch.Tensor) -> None:
    """
    Renders a horizontal bar chart showing prediction confidence for each class.
    """
    prob_values = probs.cpu().numpy()
    prob_df = pd.DataFrame({"Class": classes, "Probability": prob_values}).sort_values(
        by="Probability"
    )

    fig = px.bar(
        prob_df,
        x="Probability",
        y="Class",
        orientation="h",
        height=350,
        color="Probability",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False, xaxis=dict(range=[0, 1.1])
    )
    st.plotly_chart(fig, width="stretch")


def render_prediction_box(config: Dict[str, Any], pred_idx: int, conf: float, true_label: Optional[str] = None) -> None:
    """
    Renders prediction results in a stable container to prevent UI ghosting.
    """
    class_name = config["classes"][pred_idx]
    
    # Creating a clean container for results
    with st.container():
        if true_label is not None:
            if class_name == true_label:
                st.success(f"**Prediction:** {class_name.upper()} (Correct)")
                # Adding a blank space to ensure consistent row count
                st.write("") 
            else:
                st.error(f"**Prediction:** {class_name.upper()} (Incorrect)")
                st.info(f"**True Category:** {true_label}")
        else:
            st.success(f"**Predicted Class:** {class_name.upper()}")
            st.write("") # Spacer

        st.write(f"**Model Confidence:** {conf:.2%}")


def create_report_image(
    img: Image.Image,
    explanation_img: Optional[np.ndarray],
    class_name: str,
    conf: float,
    run_name: str,
    model_info: Dict[str, Any],
    method_name: str,
    prob_df: pd.DataFrame,
) -> bytes:
    """
    Generates a consolidated PNG report containing input, explanation,
    probabilities, and model metadata.

    Returns:
        bytes: PNG image data ready for download.
    """
    # Initialize high-quality figure
    fig = plt.figure(figsize=(15, 9), facecolor="#ffffff")
    grid = plt.GridSpec(2, 3, wspace=0.3, hspace=0.4)

    # 1. Source Image
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.imshow(img)
    ax1.set_title("Input Sample", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # 2. XAI Explanation (Grad-CAM or LIME)
    ax2 = fig.add_subplot(grid[0, 1])
    if explanation_img is not None:
        # Normalize and convert to uint8 if necessary
        if explanation_img.dtype != np.uint8:
            explanation_img = (explanation_img * 255).astype(np.uint8)
        ax2.imshow(explanation_img)
    else:
        ax2.text(0.5, 0.5, "Visualization Failed", ha="center", va="center")

    ax2.set_title(f"Explanation: {method_name}", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # 3. Probability Distribution Plot
    ax3 = fig.add_subplot(grid[:, 2])
    # Create a gradient effect for bars
    colors = plt.get_cmap("Blues")(np.linspace(0.3, 0.9, len(prob_df)))
    ax3.barh(prob_df["Class"], prob_df["Prob"], color=colors)
    ax3.set_title("Prediction Confidence", fontsize=12, fontweight="bold")
    ax3.set_xlim(0, 1.1)

    # Add numerical labels to bars
    for i, v in enumerate(prob_df["Prob"]):
        ax3.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9)

    # 4. Text Metadata Section
    ax_text = fig.add_subplot(grid[1, 0:2])
    ax_text.axis("off")

    arch_desc = model_info.get("description", "N/A")
    wrapped_arch = textwrap.fill(f"Architecture: {arch_desc}", width=70)

    report_info = (
        f"CIFAR-10 INTERPRETABILITY STUDIO REPORT\n"
        f"{'='*45}\n"
        f"Model Version: {run_name[:60]}\n\n"
        f"{wrapped_arch}\n\n"
        f"FINAL PREDICTION: {class_name.upper()}\n"
        f"CONFIDENCE SCORE: {conf:.2%}\n"
        f"INTERPRETATION: {method_name.upper()}\n"
        f"{'='*45}"
    )
    ax_text.text(
        0,
        1.0,
        report_info,
        fontsize=11,
        family="monospace",
        verticalalignment="top",
        linespacing=1.6,
    )

    # Export to memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    return buf.getvalue()
