"""
Visualization Utilities for Traffic Prediction
Author: Tulsi
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
import config

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_predictions(y_true, y_pred, title="Predictions vs Actual",
                     save_path=None, show_first_n=None):
    """
    Plot predictions vs actual values

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save figure
        show_first_n: Show only first N points
    """
    if show_first_n:
        y_true = y_true[:show_first_n]
        y_pred = y_pred[:show_first_n]

    plt.figure(figsize=(14, 6))

    plt.plot(y_true, label='Actual', color='blue', alpha=0.7, linewidth=2)
    plt.plot(y_pred, label='Predicted', color='red', alpha=0.7, linewidth=2)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Traffic Volume (vehicles/hour)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")

    plt.show()


def plot_scatter(y_true, y_pred, title="Actual vs Predicted", save_path=None):
    """
    Scatter plot of actual vs predicted
    """
    plt.figure(figsize=(8, 8))

    plt.scatter(y_true, y_pred, alpha=0.5, s=30)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val],
             'r--', linewidth=2, label='Perfect Prediction')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Actual Traffic Volume', fontsize=12)
    plt.ylabel('Predicted Traffic Volume', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Scatter plot saved to: {save_path}")

    plt.show()


def plot_residuals(y_true, y_pred, title="Residual Plot", save_path=None):
    """
    Plot residuals (errors)
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals over time
    axes[0].plot(residuals, alpha=0.7)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_title('Residuals Over Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (hours)', fontsize=12)
    axes[0].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Residual histogram
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Residual', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Residual plot saved to: {save_path}")

    plt.show()


def plot_feature_importance(feature_names, importance_values,
                            title="Feature Importance", top_n=15, save_path=None):
    """
    Plot feature importance (for tree-based models)
    """
    # Sort by importance
    indices = np.argsort(importance_values)[-top_n:]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance_values[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature importance plot saved to: {save_path}")

    plt.show()


def plot_training_history(history, title="Training History", save_path=None):
    """
    Plot training history (for neural networks)

    Args:
        history: Dictionary with 'train_loss' and 'val_loss' keys
    """
    plt.figure(figsize=(10, 6))

    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved to: {save_path}")

    plt.show()


def plot_model_comparison(results_dict, metric='MAE', save_path=None):
    """
    Bar chart comparing models

    Args:
        results_dict: Dictionary with format {model_name: metrics_dict}
        metric: Which metric to plot ('MAE', 'RMSE', 'MAPE', 'R2')
    """
    models = list(results_dict.keys())
    values = [results_dict[m][metric] for m in models]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color='steelblue', alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.title(f'Model Comparison - {metric}', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved to: {save_path}")

    plt.show()


# Quick test
if __name__ == "__main__":
    # Generate dummy data for testing
    y_true = np.random.randint(50, 500, 100)
    y_pred = y_true + np.random.normal(0, 30, 100)

    print("Testing visualization functions...\n")

    plot_predictions(y_true[:50], y_pred[:50], "Test Predictions")
    plot_scatter(y_true, y_pred, "Test Scatter")
    plot_residuals(y_true, y_pred, "Test Residuals")

    print("\n✅ All visualization functions working!")