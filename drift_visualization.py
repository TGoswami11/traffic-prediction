"""
Drift Detection Visualization for Traffic Prediction
Author: Tulsi
Created: January 2026

Purpose: Visualize drift detection results
Creates: Professional plots for monitoring model drift
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TrafficDriftVisualizer:
    """
    Create visualizations for traffic prediction drift monitoring
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer

        Args:
            output_dir: Directory to save plots (optional)
        """
        self.output_dir = output_dir
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme
        self.colors = {
            'stable': '#70AD47',  # Green
            'warning': '#FFC000',  # Orange
            'drift': '#C00000',  # Red
            'baseline': '#2E75B5'  # Blue
        }

    def plot_error_comparison(self,
                              reference_errors: np.ndarray,
                              current_errors: np.ndarray,
                              save_name: str = "error_comparison.png"):
        """
        Plot histogram comparison of error distributions

        Args:
            reference_errors: Baseline errors
            current_errors: Current errors
            save_name: Filename to save
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Reference (baseline) distribution
        ax1.hist(reference_errors, bins=30, color=self.colors['stable'],
                 alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(reference_errors), color='red',
                    linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(reference_errors):.2f}')
        ax1.set_xlabel('Prediction Error (vehicles/hour)',
                       fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Baseline Error Distribution',
                      fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Current distribution
        ax2.hist(current_errors, bins=30, color=self.colors['drift'],
                 alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(current_errors), color='blue',
                    linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(current_errors):.2f}')
        ax2.set_xlabel('Prediction Error (vehicles/hour)',
                       fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Current Error Distribution',
                      fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if self.output_dir:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.close()

    def plot_performance_timeline(self,
                                  timestamps: List[str],
                                  mae_values: List[float],
                                  drift_status: List[bool],
                                  baseline_mae: float,
                                  save_name: str = "performance_timeline.png"):
        """
        Plot MAE over time with drift regions highlighted

        Args:
            timestamps: List of timestamps
            mae_values: List of MAE values
            drift_status: List of drift detected (True/False)
            baseline_mae: Baseline MAE value
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot MAE line
        x = range(len(mae_values))
        ax.plot(x, mae_values, 'o-', linewidth=2, markersize=8,
                color=self.colors['baseline'], label='Current MAE')

        # Add baseline reference line
        ax.axhline(baseline_mae, color=self.colors['stable'],
                   linestyle='--', linewidth=2, label='Baseline MAE')

        # Highlight drift regions
        for i, drift in enumerate(drift_status):
            if drift:
                ax.axvspan(i - 0.4, i + 0.4, alpha=0.3,
                           color=self.colors['drift'])

        ax.set_xlabel('Monitoring Period', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAE (vehicles/hour)', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Timeline',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Set x-tick labels
        if len(timestamps) <= 10:
            ax.set_xticks(x)
            ax.set_xticklabels(timestamps, rotation=45, ha='right')

        plt.tight_layout()

        if self.output_dir:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.close()

    def plot_feature_drift_heatmap(self,
                                   feature_results: Dict,
                                   save_name: str = "feature_drift.png"):
        """
        Plot heatmap showing which features drifted

        Args:
            feature_results: Results from detect_data_drift()
            save_name: Filename to save
        """
        # Extract data
        features = list(feature_results.keys())
        p_values = [feature_results[f]['p_value'] for f in features]
        mean_shifts = [abs(feature_results[f]['mean_shift_percent'])
                       for f in features]

        # Determine colors based on status
        colors = []
        for f in features:
            status = feature_results[f]['status']
            if status == 'DRIFT':
                colors.append(self.colors['drift'])
            elif status == 'WARNING':
                colors.append(self.colors['warning'])
            else:
                colors.append(self.colors['stable'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(features) * 0.4)))

        # Plot 1: P-values
        bars1 = ax1.barh(features, p_values, color=colors,
                         edgecolor='black', linewidth=1.5)
        ax1.axvline(0.05, color='orange', linestyle='--',
                    linewidth=2, label='Warning (0.05)')
        ax1.axvline(0.01, color='red', linestyle='--',
                    linewidth=2, label='Drift (0.01)')
        ax1.set_xlabel('P-value', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax1.set_title('Feature Drift P-values\n(Lower = More Drift)',
                      fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)

        # Plot 2: Mean shifts
        bars2 = ax2.barh(features, mean_shifts, color=colors,
                         edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Absolute Mean Shift (%)',
                       fontsize=12, fontweight='bold')
        ax2.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax2.set_title('Feature Distribution Shifts',
                      fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if self.output_dir:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.close()

    def plot_drift_dashboard(self,
                             monitoring_log: List[Dict],
                             baseline_mae: float,
                             save_name: str = "drift_dashboard.png"):
        """
        Create comprehensive 4-panel drift monitoring dashboard

        Args:
            monitoring_log: List of monitoring results
            baseline_mae: Baseline MAE value
            save_name: Filename to save
        """
        if not monitoring_log:
            print("No monitoring data to plot")
            return

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Extract data
        timestamps = [entry.get('timestamp', f"Period {i}")
                      for i, entry in enumerate(monitoring_log)]
        mae_values = [entry['mae'] for entry in monitoring_log]
        rmse_values = [entry['rmse'] for entry in monitoring_log]
        r2_values = [entry.get('r2', 0) for entry in monitoring_log]
        drift_detected = [entry['drift_detected'] for entry in monitoring_log]

        # Panel 1: MAE Timeline (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        x = range(len(mae_values))
        ax1.plot(x, mae_values, 'o-', linewidth=2, markersize=8,
                 color=self.colors['baseline'], label='MAE')
        ax1.axhline(baseline_mae, color=self.colors['stable'],
                    linestyle='--', linewidth=2, label='Baseline')

        # Highlight drift periods
        for i, drift in enumerate(drift_detected):
            if drift:
                ax1.axvspan(i - 0.3, i + 0.3, alpha=0.3,
                            color=self.colors['drift'])

        ax1.set_ylabel('MAE (vehicles/hour)', fontsize=12, fontweight='bold')
        ax1.set_title('Performance Monitoring Timeline',
                      fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: R² Timeline
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(x, r2_values, 's-', linewidth=2, markersize=8,
                 color=self.colors['stable'])
        ax2.axhline(0, color='red', linestyle='--', linewidth=1)
        ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax2.set_title('Model Accuracy (R²)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Panel 3: RMSE Timeline
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(x, rmse_values, '^-', linewidth=2, markersize=8,
                 color='#ED7D31')
        ax3.set_ylabel('RMSE (vehicles/hour)', fontsize=12, fontweight='bold')
        ax3.set_title('Root Mean Square Error', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Panel 4: Drift Status Summary
        ax4 = fig.add_subplot(gs[2, :])

        drift_count = sum(drift_detected)
        stable_count = len(monitoring_log) - drift_count

        categories = ['Stable', 'Drift Detected']
        counts = [stable_count, drift_count]
        colors_bar = [self.colors['stable'], self.colors['drift']]

        bars = ax4.bar(categories, counts, color=colors_bar,
                       edgecolor='black', linewidth=2)
        ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax4.set_title('Drift Detection Summary', fontsize=13, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}',
                     ha='center', va='bottom', fontweight='bold', fontsize=12)

        plt.suptitle('Traffic Prediction Drift Monitoring Dashboard',
                     fontsize=16, fontweight='bold', y=0.995)

        if self.output_dir:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.close()

    def plot_mae_comparison(self,
                            baseline_mae: float,
                            current_mae: float,
                            mae_change_percent: float,
                            save_name: str = "mae_comparison.png"):
        """
        Plot simple MAE comparison bar chart

        Args:
            baseline_mae: Baseline MAE
            current_mae: Current MAE
            mae_change_percent: Percentage change
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ['Baseline\n(Validation)', 'Current\n(Recent Data)']
        values = [baseline_mae, current_mae]

        # Color based on change
        colors_bar = [self.colors['stable'],
                      self.colors['drift'] if mae_change_percent > 25
                      else self.colors['warning'] if mae_change_percent > 10
                      else self.colors['stable']]

        bars = ax.bar(categories, values, color=colors_bar,
                      edgecolor='black', linewidth=2, width=0.6)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=14)

        # Add change annotation
        ax.annotate(f'{mae_change_percent:+.1f}%',
                    xy=(1, current_mae), xytext=(1.3, current_mae),
                    fontsize=16, fontweight='bold',
                    color='red' if mae_change_percent > 0 else 'green',
                    arrowprops=dict(arrowstyle='->', color='red' if mae_change_percent > 0 else 'green'))

        ax.set_ylabel('MAE (vehicles/hour)', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance: Baseline vs Current',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if self.output_dir:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("DRIFT VISUALIZATION MODULE")
    print("=" * 70)
    print("\nUsage:")
    print("from drift_visualization import TrafficDriftVisualizer")
    print("\nviz = TrafficDriftVisualizer(output_dir='results/drift')")
    print("viz.plot_error_comparison(ref_errors, current_errors)")
    print("\n✓ Module loaded successfully!")