"""
Custom Metrics for Walmart Retail Forecasting
Implements Weighted Mean Absolute Error (WMAE) and other evaluation metrics
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logger = logging.getLogger(__name__)


def wmae(y_true, y_pred, weights):
    """
    Core Weighted Mean Absolute Error computation.

    This is the single source of truth for the WMAE formula. Both
    ``calculate_wmae`` (holiday-based API) and ``wmae_lgb_metric``
    (LightGBM callback API) delegate here.

    Args:
        y_true (array-like): Actual values.
        y_pred (array-like): Predicted values.
        weights (array-like): Non-negative per-sample weights.
            Samples with weight 0 contribute nothing to numerator or
            denominator and are effectively ignored.

    Returns:
        float: WMAE = sum(w_i * |y_i - ŷ_i|) / sum(w_i)
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    weights = np.array(weights, dtype=float)
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))


def calculate_wmae(y_true, y_pred, is_holiday):
    """
    Calculate Weighted Mean Absolute Error with holiday-based weighting.

    Convenience wrapper around :func:`wmae` that converts a boolean holiday
    flag into per-sample weights (5 for holiday weeks, 1 for non-holiday).

    Args:
        y_true (array-like): Actual sales values.
        y_pred (array-like): Predicted sales values.
        is_holiday (array-like): Boolean array indicating holiday weeks.

    Returns:
        float: WMAE score.
    """
    weights = np.where(np.array(is_holiday, dtype=bool), 5.0, 1.0)
    return wmae(y_true, y_pred, weights)


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error
    
    Args:
        y_true: Actual sales values
        y_pred: Predicted sales values
    
    Returns:
        float: MAE score
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error
    
    Args:
        y_true: Actual sales values
        y_pred: Predicted sales values
    
    Returns:
        float: RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error
    
    Args:
        y_true: Actual sales values
        y_pred: Predicted sales values
    
    Returns:
        float: MAPE score (as percentage)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return mape


def evaluate_model(y_true, y_pred, is_holiday):
    """
    Calculate all evaluation metrics
    
    Args:
        y_true: Actual sales values
        y_pred: Predicted sales values
        is_holiday: Boolean array indicating holiday weeks
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'wmae': calculate_wmae(y_true, y_pred, is_holiday),
        'mae': calculate_mae(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred)
    }
    
    logger.info(f"Model Evaluation Metrics:")
    logger.info(f"  WMAE: {metrics['wmae']:.2f}")
    logger.info(f"  MAE:  {metrics['mae']:.2f}")
    logger.info(f"  RMSE: {metrics['rmse']:.2f}")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    
    return metrics


def wmae_lgb_metric(y_pred, dtrain):
    """
    LightGBM-compatible wrapper for :func:`wmae`.

    LightGBM passes ``(y_pred, dtrain)`` and expects
    ``(metric_name, value, is_higher_better)`` in return.

    Args:
        y_pred (array-like): Predicted values supplied by LightGBM.
        dtrain: LightGBM Dataset object. Holiday weights are read from the
            ``'is_holiday'`` field if it was set on the dataset; otherwise
            every sample is treated as a non-holiday (weight = 1).

    Returns:
        tuple: ('wmae', float, False)
    """
    y_true = dtrain.get_label()
    is_holiday = dtrain.get_field('is_holiday')
    if is_holiday is None:
        is_holiday = np.zeros(len(y_true), dtype=bool)
    weights = np.where(np.array(is_holiday, dtype=bool), 5.0, 1.0)
    return 'wmae', wmae(y_true, y_pred, weights), False
