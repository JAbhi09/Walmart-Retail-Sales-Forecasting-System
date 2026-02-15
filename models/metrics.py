"""
Custom Metrics for Walmart Retail Forecasting
Implements Weighted Mean Absolute Error (WMAE) and other evaluation metrics
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logger = logging.getLogger(__name__)


def calculate_wmae(y_true, y_pred, is_holiday):
    """
    Calculate Weighted Mean Absolute Error (WMAE)
    
    WMAE is the primary evaluation metric for Walmart recruiting competition.
    Holiday weeks are weighted 5x more than non-holiday weeks.
    
    Args:
        y_true: Actual sales values
        y_pred: Predicted sales values
        is_holiday: Boolean array indicating holiday weeks
    
    Returns:
        float: WMAE score
    
    Formula:
        WMAE = sum(w_i * |y_i - ŷ_i|) / sum(w_i)
        where w_i = 5 if is_holiday else 1
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    is_holiday = np.array(is_holiday)
    
    # Calculate weights (5 for holiday, 1 for non-holiday)
    weights = np.where(is_holiday, 5.0, 1.0)
    
    # Calculate weighted absolute errors
    absolute_errors = np.abs(y_true - y_pred)
    weighted_errors = weights * absolute_errors
    
    # Calculate WMAE
    wmae = np.sum(weighted_errors) / np.sum(weights)
    
    return wmae


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
    Custom WMAE metric for LightGBM
    
    LightGBM requires metrics in format: (metric_name, metric_value, is_higher_better)
    
    Args:
        y_pred: Predicted values
        dtrain: LightGBM Dataset object
    
    Returns:
        tuple: (metric_name, metric_value, is_higher_better)
    """
    y_true = dtrain.get_label()
    
    # Get is_holiday from dataset metadata
    # Note: This needs to be set when creating the dataset
    is_holiday = dtrain.get_field('is_holiday')
    if is_holiday is None:
        # Fallback: treat all as non-holiday
        is_holiday = np.zeros(len(y_true), dtype=bool)
    
    wmae = calculate_wmae(y_true, y_pred, is_holiday)
    
    # Return (name, value, is_higher_better)
    # Lower WMAE is better, so is_higher_better=False
    return 'wmae', wmae, False
