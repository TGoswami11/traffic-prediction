"""
Drift Detection Metrics for Traffic Prediction
Author: Tulsi
Created: January 2026

Purpose: Detect when traffic prediction model performance degrades
Uses: Statistical tests to monitor model drift
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional


class TrafficDriftMetrics:
    """
    Drift detection metrics specifically for traffic prediction

    Detects 3 types of drift:
    1. Performance Drift - Model accuracy degrades
    2. Data Drift - Feature distributions change
    3. Statistical Drift - Error patterns change
    """

    def __init__(self,
                 warning_threshold: float = 0.05,  # 95% confidence
                 drift_threshold: float = 0.01):  # 99% confidence
        """
        Initialize drift metrics

        Args:
            warning_threshold: p-value for warning level
            drift_threshold: p-value for drift detection
        """
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold

        # Storage for reference (baseline) data
        self.reference_errors = None
        self.reference_mae = None
        self.reference_rmse = None
        self.reference_features = {}

        print(f"✓ Drift metrics initialized")
        print(f"  Warning threshold: {warning_threshold} (95% confidence)")
        print(f"  Drift threshold: {drift_threshold} (99% confidence)")

    def set_baseline(self,
                     y_true: np.ndarray,
                     y_pred: np.ndarray,
                     features: Optional[pd.DataFrame] = None):
        """
        Set baseline (reference) data from validation set

        Args:
            y_true: True traffic values from validation
            y_pred: Predicted traffic values from validation
            features: Feature DataFrame (optional for data drift)
        """
        # Calculate baseline errors
        self.reference_errors = np.abs(y_true - y_pred)
        self.reference_mae = np.mean(self.reference_errors)
        self.reference_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        print(f"\n✓ Baseline set from {len(y_true)} samples")
        print(f"  Reference MAE:  {self.reference_mae:.2f} vehicles/hour")
        print(f"  Reference RMSE: {self.reference_rmse:.2f} vehicles/hour")

        # Store feature distributions if provided
        if features is not None:
            for col in features.columns:
                self.reference_features[col] = {
                    'mean': features[col].mean(),
                    'std': features[col].std(),
                    'values': features[col].values.copy()
                }
            print(f"  Features tracked: {len(self.reference_features)}")

    def detect_performance_drift(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray) -> Dict:
        """
        Detect performance drift using statistical tests

        Args:
            y_true: Current true values
            y_pred: Current predictions

        Returns:
            Dictionary with drift detection results
        """
        if self.reference_errors is None:
            raise ValueError("Baseline not set! Call set_baseline() first.")

        # Calculate current errors
        current_errors = np.abs(y_true - y_pred)
        current_mae = np.mean(current_errors)
        current_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # Test 1: Kolmogorov-Smirnov Test (distribution comparison)
        ks_stat, ks_pvalue = stats.ks_2samp(
            self.reference_errors,
            current_errors
        )

        # Test 2: Mann-Whitney U Test (median comparison)
        mw_stat, mw_pvalue = stats.mannwhitneyu(
            self.reference_errors,
            current_errors,
            alternative='two-sided'
        )

        # Test 3: Calculate performance change
        mae_change = ((current_mae - self.reference_mae) / self.reference_mae) * 100
        rmse_change = ((current_rmse - self.reference_rmse) / self.reference_rmse) * 100

        # Determine drift status
        drift_detected = False
        warning_detected = False

        if ks_pvalue < self.drift_threshold or mw_pvalue < self.drift_threshold:
            drift_detected = True
            status = "🚨 DRIFT DETECTED"
        elif ks_pvalue < self.warning_threshold or mw_pvalue < self.warning_threshold:
            warning_detected = True
            status = "⚠️  WARNING"
        else:
            status = "✓ STABLE"

        results = {
            'status': status,
            'drift_detected': drift_detected,
            'warning_detected': warning_detected,
            'current_mae': current_mae,
            'reference_mae': self.reference_mae,
            'mae_change_percent': mae_change,
            'current_rmse': current_rmse,
            'reference_rmse': self.reference_rmse,
            'rmse_change_percent': rmse_change,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'mw_statistic': mw_stat,
            'mw_pvalue': mw_pvalue,
            'samples': len(current_errors)
        }

        return results

    def detect_data_drift(self,
                          current_features: pd.DataFrame) -> Dict:
        """
        Detect drift in feature distributions

        Args:
            current_features: Current feature DataFrame

        Returns:
            Dictionary with per-feature drift results
        """
        if not self.reference_features:
            raise ValueError("Baseline features not set!")

        feature_results = {}
        drifted_features = []
        warning_features = []

        for feature_name in current_features.columns:
            if feature_name not in self.reference_features:
                continue

            # Get reference and current values
            ref_values = self.reference_features[feature_name]['values']
            curr_values = current_features[feature_name].values

            # Kolmogorov-Smirnov test for distribution
            ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)

            # Calculate statistics
            ref_mean = self.reference_features[feature_name]['mean']
            curr_mean = current_features[feature_name].mean()
            mean_shift = curr_mean - ref_mean
            mean_shift_pct = (mean_shift / ref_mean * 100) if ref_mean != 0 else 0

            # Determine status
            if p_value < self.drift_threshold:
                status = 'DRIFT'
                drifted_features.append(feature_name)
            elif p_value < self.warning_threshold:
                status = 'WARNING'
                warning_features.append(feature_name)
            else:
                status = 'STABLE'

            feature_results[feature_name] = {
                'status': status,
                'p_value': p_value,
                'ks_statistic': ks_stat,
                'ref_mean': ref_mean,
                'curr_mean': curr_mean,
                'mean_shift': mean_shift,
                'mean_shift_percent': mean_shift_pct
            }

        summary = {
            'total_features': len(feature_results),
            'drifted_count': len(drifted_features),
            'warning_count': len(warning_features),
            'stable_count': len(feature_results) - len(drifted_features) - len(warning_features),
            'drifted_features': drifted_features,
            'warning_features': warning_features,
            'feature_results': feature_results
        }

        return summary

    def get_drift_severity(self,
                           performance_drift: Dict,
                           data_drift: Optional[Dict] = None) -> str:
        """
        Calculate overall drift severity

        Args:
            performance_drift: Results from detect_performance_drift()
            data_drift: Results from detect_data_drift() (optional)

        Returns:
            Severity level: CRITICAL, HIGH, MEDIUM, LOW, NONE
        """
        severity_score = 0

        # Performance drift scoring
        if performance_drift['drift_detected']:
            severity_score += 3
        elif performance_drift['warning_detected']:
            severity_score += 1

        # MAE change scoring
        mae_change = abs(performance_drift['mae_change_percent'])
        if mae_change > 50:
            severity_score += 3
        elif mae_change > 25:
            severity_score += 2
        elif mae_change > 10:
            severity_score += 1

        # Data drift scoring (if available)
        if data_drift:
            drift_ratio = data_drift['drifted_count'] / data_drift['total_features']
            if drift_ratio > 0.5:
                severity_score += 2
            elif drift_ratio > 0.25:
                severity_score += 1

        # Determine severity level
        if severity_score >= 6:
            return "CRITICAL"
        elif severity_score >= 4:
            return "HIGH"
        elif severity_score >= 2:
            return "MEDIUM"
        elif severity_score >= 1:
            return "LOW"
        else:
            return "NONE"

    def recommend_action(self, severity: str) -> str:
        """
        Recommend action based on drift severity

        Args:
            severity: Severity level from get_drift_severity()

        Returns:
            Recommended action string
        """
        actions = {
            "CRITICAL": "🚨 IMMEDIATE ACTION: Retrain model with recent data (last 30 days)",
            "HIGH": "⚠️  HIGH PRIORITY: Schedule retraining within 24 hours",
            "MEDIUM": "⚠️  MEDIUM: Monitor closely, prepare for retraining",
            "LOW": "ℹ️  LOW: Continue monitoring, no immediate action needed",
            "NONE": "✓ STABLE: Model performing well, routine monitoring"
        }

        return actions.get(severity, "Unknown severity")


def calculate_traffic_metrics(y_true: np.ndarray,
                              y_pred: np.ndarray) -> Dict:
    """
    Calculate traffic prediction metrics

    Args:
        y_true: True traffic values
        y_pred: Predicted traffic values

    Returns:
        Dictionary with MAE, RMSE, R2, MAPE
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


def print_drift_report(performance_drift: Dict,
                       data_drift: Optional[Dict] = None):
    """
    Print formatted drift detection report

    Args:
        performance_drift: Results from detect_performance_drift()
        data_drift: Results from detect_data_drift() (optional)
    """
    print("\n" + "=" * 70)
    print("DRIFT DETECTION REPORT")
    print("=" * 70)

    # Performance drift
    print(f"\n📊 PERFORMANCE DRIFT: {performance_drift['status']}")
    print(f"   Current MAE:  {performance_drift['current_mae']:.2f} vehicles/hour")
    print(f"   Baseline MAE: {performance_drift['reference_mae']:.2f} vehicles/hour")
    print(f"   Change:       {performance_drift['mae_change_percent']:+.1f}%")
    print(f"   KS p-value:   {performance_drift['ks_pvalue']:.4f}")
    print(f"   Samples:      {performance_drift['samples']}")

    # Data drift
    if data_drift:
        print(f"\n📈 DATA DRIFT:")
        print(f"   Total Features:   {data_drift['total_features']}")
        print(f"   Drifted:          {data_drift['drifted_count']}")
        print(f"   Warning:          {data_drift['warning_count']}")
        print(f"   Stable:           {data_drift['stable_count']}")

        if data_drift['drifted_features']:
            print(f"\n   Drifted Features: {', '.join(data_drift['drifted_features'])}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("DRIFT METRICS - EXAMPLE")
    print("=" * 70)

    # Simulate baseline data (validation set)
    np.random.seed(42)
    y_val = 300 + 50 * np.sin(np.linspace(0, 10, 120)) + np.random.normal(0, 20, 120)
    y_pred_val = y_val + np.random.normal(0, 19, 120)  # MAE ~19 like XGBoost

    # Initialize metrics
    drift_metrics = TrafficDriftMetrics()
    drift_metrics.set_baseline(y_val, y_pred_val)

    # Scenario 1: Stable performance
    print("\n" + "=" * 70)
    print("SCENARIO 1: STABLE PERFORMANCE")
    print("=" * 70)
    y_test = 300 + 50 * np.sin(np.linspace(10, 12, 50)) + np.random.normal(0, 20, 50)
    y_pred_test = y_test + np.random.normal(0, 19, 50)

    result = drift_metrics.detect_performance_drift(y_test, y_pred_test)
    print_drift_report(result)

    # Scenario 2: Drift detected
    print("\n" + "=" * 70)
    print("SCENARIO 2: DRIFT DETECTED")
    print("=" * 70)
    y_test_drift = 300 + 50 * np.sin(np.linspace(12, 14, 50)) + np.random.normal(0, 20, 50)
    y_pred_drift = y_test_drift + np.random.normal(0, 60, 50)  # Much worse!

    result_drift = drift_metrics.detect_performance_drift(y_test_drift, y_pred_drift)
    print_drift_report(result_drift)

    # Get severity and recommendation
    severity = drift_metrics.get_drift_severity(result_drift)
    action = drift_metrics.recommend_action(severity)

    print(f"\n🎯 SEVERITY: {severity}")
    print(f"💡 RECOMMENDATION: {action}")

    print("\n✓ Example complete!")