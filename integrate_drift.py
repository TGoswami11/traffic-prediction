"""
Drift Detection Integration - January 2023 Traffic Project
Author: Tulsi
Created: January 2026

Purpose: Integrate drift detection into existing traffic prediction project
Usage: Run this after training XGBoost model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import project modules
import config
from src.models.xgboost_model import XGBoostTrafficModel

# Import drift detection modules
from drift_metrics import TrafficDriftMetrics, print_drift_report
from drift_visualization import TrafficDriftVisualizer


def integrate_drift_detection():
    """
    Complete drift detection integration example

    Demonstrates:
    1. Loading trained model
    2. Setting baseline from validation set
    3. Monitoring test set for drift
    4. Creating visualizations
    5. Generating recommendations
    """

    print("\n" + "=" * 80)
    print("DRIFT DETECTION INTEGRATION - JANUARY 2023 PROJECT")
    print("=" * 80)

    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n[Step 1] Loading featured data...")
    data_file = Path(config.PROCESSED_DATA_DIR) / "traffic_2023_01_featured.csv"
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✓ Loaded {len(df)} records")

    # Split data
    train_df = df[(df['timestamp'] >= config.TRAIN_START) &
                  (df['timestamp'] <= config.TRAIN_END)]
    val_df = df[(df['timestamp'] >= config.VAL_START) &
                (df['timestamp'] <= config.VAL_END)]
    test_df = df[(df['timestamp'] >= config.TEST_START) &
                 (df['timestamp'] <= config.TEST_END)]

    print(f"  Training:   {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test:       {len(test_df)} samples")

    # ========================================================================
    # STEP 2: Load Trained Model
    # ========================================================================
    print("\n[Step 2] Loading trained XGBoost model...")
    model = XGBoostTrafficModel()
    model_path = Path(config.MODEL_DIR) / "xgboost_jan2023.pkl"

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Please run train_xgboost.py first!")
        return

    model.load_model(model_path)
    print(f"✓ Model loaded from: {model_path}")

    # ========================================================================
    # STEP 3: Initialize Drift Detection
    # ========================================================================
    print("\n[Step 3] Initializing drift detection...")
    drift_metrics = TrafficDriftMetrics(
        warning_threshold=0.05,  # 95% confidence
        drift_threshold=0.01  # 99% confidence
    )

    # Create output directory
    drift_dir = Path(config.RESULTS_DIR) / "drift"
    drift_dir.mkdir(parents=True, exist_ok=True)

    visualizer = TrafficDriftVisualizer(output_dir=drift_dir)
    print(f"✓ Visualizations will be saved to: {drift_dir}")

    # ========================================================================
    # STEP 4: Set Baseline from Validation Set
    # ========================================================================
    print("\n[Step 4] Setting baseline from validation set...")

    # Prepare validation data
    X_val, y_val = model.prepare_data(val_df)
    y_pred_val = model.predict(X_val)

    # Set baseline
    drift_metrics.set_baseline(y_val, y_pred_val, val_df)

    # Store reference errors for visualization
    reference_errors = np.abs(y_val - y_pred_val)

    # ========================================================================
    # STEP 5: Monitor Test Set for Drift
    # ========================================================================
    print("\n[Step 5] Monitoring test set for drift...")

    # Prepare test data
    X_test, y_test = model.prepare_data(test_df)
    y_pred_test = model.predict(X_test)

    # Detect performance drift
    perf_drift = drift_metrics.detect_performance_drift(y_test, y_pred_test)

    # Detect data drift
    # Get features only (exclude non-feature columns)
    exclude_cols = ['timestamp', 'PKW', 'LKW', 'Buses', 'Total']
    val_features = val_df[[col for col in val_df.columns
                           if col not in exclude_cols]]
    test_features = test_df[[col for col in test_df.columns
                             if col not in exclude_cols]]

    data_drift = drift_metrics.detect_data_drift(test_features)

    # Print report
    print_drift_report(perf_drift, data_drift)

    # ========================================================================
    # STEP 6: Assess Severity and Recommendations
    # ========================================================================
    print("\n[Step 6] Assessing drift severity...")

    severity = drift_metrics.get_drift_severity(perf_drift, data_drift)
    recommendation = drift_metrics.recommend_action(severity)

    print(f"\n🎯 DRIFT SEVERITY: {severity}")
    print(f"💡 RECOMMENDATION: {recommendation}")

    # ========================================================================
    # STEP 7: Create Visualizations
    # ========================================================================
    print("\n[Step 7] Creating visualizations...")

    # Plot 1: Error distribution comparison
    current_errors = np.abs(y_test - y_pred_test)
    visualizer.plot_error_comparison(
        reference_errors,
        current_errors,
        save_name="01_error_comparison.png"
    )

    # Plot 2: MAE comparison
    visualizer.plot_mae_comparison(
        baseline_mae=perf_drift['reference_mae'],
        current_mae=perf_drift['current_mae'],
        mae_change_percent=perf_drift['mae_change_percent'],
        save_name="02_mae_comparison.png"
    )

    # Plot 3: Feature drift heatmap
    visualizer.plot_feature_drift_heatmap(
        data_drift['feature_results'],
        save_name="03_feature_drift.png"
    )

    print(f"\n✓ All visualizations saved to: {drift_dir}")

    # ========================================================================
    # STEP 8: Save Drift Report
    # ========================================================================
    print("\n[Step 8] Saving drift report...")

    report_path = drift_dir / "drift_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DRIFT DETECTION REPORT - JANUARY 2023 PROJECT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Model: XGBoost Traffic Predictor\n")
        f.write(f"Baseline: Validation Set (Jan 22-26, {len(y_val)} samples)\n")
        f.write(f"Monitored: Test Set (Jan 27-31, {len(y_test)} samples)\n\n")

        f.write("=" * 80 + "\n")
        f.write("PERFORMANCE DRIFT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Status: {perf_drift['status']}\n")
        f.write(f"Baseline MAE: {perf_drift['reference_mae']:.2f} vehicles/hour\n")
        f.write(f"Current MAE:  {perf_drift['current_mae']:.2f} vehicles/hour\n")
        f.write(f"Change:       {perf_drift['mae_change_percent']:+.1f}%\n")
        f.write(f"KS p-value:   {perf_drift['ks_pvalue']:.4f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("DATA DRIFT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Features:   {data_drift['total_features']}\n")
        f.write(f"Drifted:          {data_drift['drifted_count']}\n")
        f.write(f"Warning:          {data_drift['warning_count']}\n")
        f.write(f"Stable:           {data_drift['stable_count']}\n\n")

        if data_drift['drifted_features']:
            f.write(f"Drifted Features: {', '.join(data_drift['drifted_features'])}\n\n")

        f.write("=" * 80 + "\n")
        f.write("ASSESSMENT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Severity: {severity}\n")
        f.write(f"Recommendation: {recommendation}\n\n")

        f.write("=" * 80 + "\n")
        f.write("VISUALIZATIONS\n")
        f.write("=" * 80 + "\n")
        f.write(f"1. Error distribution comparison\n")
        f.write(f"2. MAE comparison chart\n")
        f.write(f"3. Feature drift heatmap\n\n")

        f.write(f"All files saved to: {drift_dir}\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Report saved to: {report_path}")

    # ========================================================================
    # STEP 9: Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("DRIFT DETECTION COMPLETE!")
    print("=" * 80)

    print(f"\n📊 Results Summary:")
    print(f"   Performance Drift: {perf_drift['status']}")
    print(f"   Data Drift: {data_drift['drifted_count']} / {data_drift['total_features']} features")
    print(f"   Severity: {severity}")

    print(f"\n📁 Output Files:")
    print(f"   {drift_dir}/01_error_comparison.png")
    print(f"   {drift_dir}/02_mae_comparison.png")
    print(f"   {drift_dir}/03_feature_drift.png")
    print(f"   {drift_dir}/drift_report.txt")

    print(f"\n💡 Next Steps:")
    if severity in ['CRITICAL', 'HIGH']:
        print(f"   🚨 Retrain model immediately with recent data!")
        print(f"   📝 Use last 30 days for retraining")
    elif severity == 'MEDIUM':
        print(f"   ⚠️  Monitor closely, prepare retraining pipeline")
    else:
        print(f"   ✓ Model stable, continue routine monitoring")

    print("\n" + "=" * 80)


def simulate_future_drift():
    """
    Simulate drift detection for future data (February 2023)

    Demonstrates what happens when traffic patterns change
    """
    print("\n" + "=" * 80)
    print("SIMULATING FUTURE DRIFT SCENARIO")
    print("=" * 80)
    print("\nScenario: Traffic patterns change in February 2023")
    print("           (e.g., new road construction, holiday effect)")

    # Load model and validation data
    print("\n[1] Loading model and baseline...")
    model = XGBoostTrafficModel()
    model_path = Path(config.MODEL_DIR) / "xgboost_jan2023.pkl"
    model.load_model(model_path)

    # Load data
    data_file = Path(config.PROCESSED_DATA_DIR) / "traffic_2023_01_featured.csv"
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    val_df = df[(df['timestamp'] >= config.VAL_START) &
                (df['timestamp'] <= config.VAL_END)]

    # Initialize drift detection
    drift_metrics = TrafficDriftMetrics()
    X_val, y_val = model.prepare_data(val_df)
    y_pred_val = model.predict(X_val)
    drift_metrics.set_baseline(y_val, y_pred_val)

    # Simulate drifted data (e.g., traffic increases by 30%)
    print("\n[2] Simulating drifted data (traffic +30%)...")
    test_df = df[(df['timestamp'] >= config.TEST_START) &
                 (df['timestamp'] <= config.TEST_END)]

    X_test, y_test_original = model.prepare_data(test_df)

    # Simulate drift: increase actual traffic by 30%
    y_test_drifted = y_test_original * 1.3

    # Model predictions stay same (model doesn't know about drift)
    y_pred_test = model.predict(X_test)

    # Detect drift
    print("\n[3] Detecting drift...")
    perf_drift = drift_metrics.detect_performance_drift(
        y_test_drifted,
        y_pred_test
    )

    print_drift_report(perf_drift)

    severity = drift_metrics.get_drift_severity(perf_drift)
    recommendation = drift_metrics.recommend_action(severity)

    print(f"\n🎯 DRIFT SEVERITY: {severity}")
    print(f"💡 RECOMMENDATION: {recommendation}")

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    # Run drift detection integration
    try:
        integrate_drift_detection()

        # Optionally run simulation
        print("\n" + "=" * 80)
        response = input("Run future drift simulation? (y/n): ")
        if response.lower() == 'y':
            simulate_future_drift()

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Trained XGBoost model (run train_xgboost.py)")
        print("  2. Featured data in processed/ directory")
        print("  3. Correct project structure")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()