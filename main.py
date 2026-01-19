"""
Traffic Prediction - Main Script
Author: Tulsi
Thesis: Master Elektro- und Informationstechnik, THD
"""

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config


def test_project_setup():
    """Test that project is set up correctly"""

    print("=" * 80)
    print("TRAFFIC PREDICTION PROJECT - SETUP TEST")
    print("=" * 80)

    # Test 1: Configuration
    print("\n Test 1: Configuration")
    print(f"   Project Root: {config.PROJECT_ROOT}")
    print(f"   Data Directory: {config.DATA_DIR}")
    print(f"   Models Directory: {config.MODEL_DIR}")

    # Test 2: Import modules
    print("\n Test 2: Import Core Libraries")
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        import xgboost
        import torch
        import statsmodels
        print("   All core libraries imported successfully!")
    except ImportError as e:
        print(f"  Error importing: {e}")
        return

    # Test 3: Generate sample data
    print("\n Test 3: Generate Sample Traffic Data")

    np.random.seed(42)

    # Create 1 year of hourly data
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(24 * 365)]

    # Simulate realistic traffic patterns
    hours = np.array([d.hour for d in dates])
    days = np.array([d.weekday() for d in dates])

    # Base traffic with patterns
    base_traffic = 100
    hourly_pattern = 50 * np.sin(2 * np.pi * (hours - 6) / 24)  # Peak at 6am and 6pm
    weekend_effect = -30 * (days >= 5)  # Reduced traffic on weekends
    noise = np.random.normal(0, 10, len(dates))

    traffic = base_traffic + hourly_pattern + weekend_effect + noise
    traffic = np.maximum(traffic, 10)  # Minimum 10 vehicles

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'traffic_count': traffic.astype(int),
        'hour': hours,
        'day_of_week': days,
        'is_weekend': (days >= 5).astype(int)
    })

    print(f"   Generated {len(df)} hours of traffic data")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Traffic range: {df['traffic_count'].min()} to {df['traffic_count'].max()}")

    # Test 4: Save sample data
    print("\n✅ Test 4: Save Sample Data")
    output_path = config.PROCESSED_DATA_DIR + '/sample_traffic_data.csv'
    df.to_csv(output_path, index=False)
    print(f"   Saved to: {output_path}")

    # Test 5: Basic statistics
    print("\n✅ Test 5: Basic Statistics")
    print(f"   Mean traffic: {df['traffic_count'].mean():.2f}")
    print(f"   Std traffic: {df['traffic_count'].std():.2f}")
    print(f"   Weekday avg: {df[df['is_weekend'] == 0]['traffic_count'].mean():.2f}")
    print(f"   Weekend avg: {df[df['is_weekend'] == 1]['traffic_count'].mean():.2f}")

    print("\n" + "=" * 80)
    print("✅ PROJECT SETUP COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Start implementing data_loader.py")
    print("2. Then preprocessing.py")
    print("3. Then feature_engineering.py")
    print("4. Finally train your first model!")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_project_setup()
