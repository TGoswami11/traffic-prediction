"""
Evaluation Metrics for Traffic Prediction
Author: Tulsi
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mae(y_true, y_pred):
    """Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_r2(y_true, y_pred):
    """R-squared Score"""
    return r2_score(y_true, y_pred)


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Calculate all metrics and print results

    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model

    Returns:
        dict: Dictionary with all metrics
    """
    metrics = {
        'MAE': calculate_mae(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred)
    }

    print(f"\n{'=' * 60}")
    print(f"{model_name} - Evaluation Metrics")
    print(f"{'=' * 60}")
    print(f"MAE:   {metrics['MAE']:.2f}")
    print(f"RMSE:  {metrics['RMSE']:.2f}")
    print(f"MAPE:  {metrics['MAPE']:.2f}%")
    print(f"R²:    {metrics['R2']:.4f}")
    print(f"{'=' * 60}\n")

    return metrics


def compare_models(results_dict):
    """
    Compare multiple models

    Args:
        results_dict: Dictionary with format {model_name: metrics_dict}
    """
    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Model':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'R²':<10}")
    print(f"{'-' * 70}")

    for model_name, metrics in results_dict.items():
        print(f"{model_name:<15} "
              f"{metrics['MAE']:<10.2f} "
              f"{metrics['RMSE']:<10.2f} "
              f"{metrics['MAPE']:<10.2f} "
              f"{metrics['R2']:<10.4f}")

    print(f"{'=' * 70}\n")

    # Find best model
    best_model = min(results_dict.items(), key=lambda x: x[1]['MAE'])
    print(f"🏆 Best Model (by MAE): {best_model[0]}")
    print(f"   MAE: {best_model[1]['MAE']:.2f}\n")