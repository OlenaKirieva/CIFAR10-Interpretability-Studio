import logging
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def get_error_metrics(df_preds: pd.DataFrame) -> Tuple[int, float]:
    """
    Calculates overall error statistics for the provided predictions.

    Args:
        df_preds: DataFrame containing 'true_label' and 'pred_label' columns.

    Returns:
        tuple: (total_number_of_errors, accuracy_score)
    """
    total = len(df_preds)
    if total == 0:
        return 0, 0.0

    errors = df_preds[df_preds["true_label"] != df_preds["pred_label"]]
    error_count = len(errors)
    accuracy = (total - error_count) / total

    return error_count, accuracy


def get_filtered_errors(
    df_preds: pd.DataFrame,
    true_cls: str,
    pred_cls: str,
    classes: List[str],
    sort_mode: str,
) -> pd.DataFrame:
    """
    Filters the full prediction set to identify specific misclassification patterns.

    Args:
        df_preds: The full dataset of predictions.
        true_cls: The ground truth class name (or "All").
        pred_cls: The class name predicted by the model (or "All Errors").
        classes: List of all possible class names.
        sort_mode: Sorting strategy ("Highest Confidence" or "Lowest Confidence").

    Returns:
        pd.DataFrame: Sorted subset of misclassified samples.
    """
    logger.info(
        f"Applying error filters: True Class='{true_cls}', Predicted='{pred_cls}'"
    )

    # 1. Start with samples where prediction doesn't match ground truth
    errors = df_preds[df_preds["true_label"] != df_preds["pred_label"]].copy()

    # 2. Apply True Class filter
    if true_cls != "All":
        true_idx = classes.index(true_cls)
        errors = errors[errors["true_label"] == true_idx]

    # 3. Apply Predicted Class filter
    if pred_cls not in ["All Errors", "All", "Correct Only"]:
        pred_idx = classes.index(pred_cls)
        errors = errors[errors["pred_label"] == pred_idx]

    # 4. Handle Sorting
    # We use ascending=False for Highest Confidence (closer to 1.0)
    is_ascending = sort_mode == "Lowest Confidence"
    sorted_errors = errors.sort_values(by="confidence", ascending=is_ascending)

    logger.info(f"Filtering complete. Found {len(sorted_errors)} matching samples.")

    return sorted_errors
