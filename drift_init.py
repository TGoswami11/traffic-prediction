"""
Drift Detection Package for Traffic Prediction
Author: Tulsi
Created: January 2026

Package Structure:
- drift_metrics.py: Core drift detection algorithms
- drift_visualization.py: Visualization tools
- __init__.py: Package initialization (this file)

Usage:
    from drift_detection import TrafficDriftMetrics, TrafficDriftVisualizer

    # Initialize
    drift_metrics = TrafficDriftMetrics()
    visualizer = TrafficDriftVisualizer(output_dir='results/drift')

    # Set baseline from validation set
    drift_metrics.set_baseline(y_val, y_pred_val, X_val)

    # Monitor new data
    drift_result = drift_metrics.detect_performance_drift(y_test, y_pred_test)

    # Visualize
    visualizer.plot_error_comparison(ref_errors, current_errors)
"""

__version__ = '1.0.0'
__author__ = 'Tulsi Gausvami'

# Import main classes for easy access
from drift_metrics import (
    TrafficDriftMetrics,
    calculate_traffic_metrics,
    print_drift_report
)

from drift_visualization import TrafficDriftVisualizer

# Define what gets imported with "from drift_detection import *"
__all__ = [
    'TrafficDriftMetrics',
    'TrafficDriftVisualizer',
    'calculate_traffic_metrics',
    'print_drift_report'
]

# Package information
PACKAGE_INFO = {
    'name': 'drift_detection',
    'version': __version__,
    'author': __author__,
    'description': 'Drift detection for traffic prediction models',
    'components': [
        'TrafficDriftMetrics - Core drift detection',
        'TrafficDriftVisualizer - Visualization tools'
    ]
}


def print_package_info():
    """Print package information"""
    print("=" * 70)
    print(f"DRIFT DETECTION PACKAGE v{__version__}")
    print("=" * 70)
    print(f"Author: {__author__}")
    print(f"\nComponents:")
    for comp in PACKAGE_INFO['components']:
        print(f"  • {comp}")
    print("\nUsage:")
    print("  from drift_detection import TrafficDriftMetrics, TrafficDriftVisualizer")
    print("=" * 70)


# Print info when package is imported
print(f"✓ Drift Detection Package v{__version__} loaded")