"""
Feature Engineering - January 2023
Author: Tulsi
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


class FeatureEngineer:
    """Create features for traffic prediction"""

    def __init__(self, df):
        self.df = df.copy()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.target_col = 'PKW'  # Predict passenger cars

    def create_temporal_features(self):
        """Create time-based features"""
        print("\n" + "=" * 70)
        print("CREATING TEMPORAL FEATURES")
        print("=" * 70)

        # Basic temporal
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['day_of_month'] = self.df['timestamp'].dt.day

        # Cyclical encoding (important for time!)
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)

        # Binary indicators
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        self.df['is_rush_hour'] = (
                ((self.df['hour'] >= 7) & (self.df['hour'] <= 9)) |
                ((self.df['hour'] >= 17) & (self.df['hour'] <= 19))
        ).astype(int)

        print(" Created 7 temporal features")
        return self

    def create_lag_features(self):
        """Create lag features (past values)"""
        print("\n" + "=" * 70)
        print("CREATING LAG FEATURES")
        print("=" * 70)

        # Lags: 1h, 2h, 3h, 6h, 12h, 24h, 48h
        lags = [1, 2, 3, 6, 12, 24, 48]

        for lag in lags:
            col_name = f'lag_{lag}'
            self.df[col_name] = self.df[self.target_col].shift(lag)
            print(f"  ✓ {col_name}")

        print(f" Created {len(lags)} lag features")
        return self

    def create_rolling_features(self):
        """Create rolling window statistics"""
        print("\n" + "=" * 70)
        print("CREATING ROLLING FEATURES")
        print("=" * 70)

        windows = [3, 6, 12, 24]

        for w in windows:
            # Rolling mean
            self.df[f'roll_mean_{w}'] = (
                self.df[self.target_col]
                .rolling(window=w, min_periods=1)
                .mean()
            )

            # Rolling std
            self.df[f'roll_std_{w}'] = (
                self.df[self.target_col]
                .rolling(window=w, min_periods=1)
                .std()
                .fillna(0)
            )

            print(f"  ✓ {w}h window: mean, std")

        print(f" Created {len(windows) * 2} rolling features")
        return self

    def drop_na_rows(self):
        """Remove rows with NaN from lag features"""
        print("\n" + "=" * 70)
        print("CLEANING FEATURE DATA")
        print("=" * 70)

        before = len(self.df)
        self.df = self.df.dropna()
        after = len(self.df)

        print(f"Dropped {before - after} NaN rows")
        print(f" Final: {after} rows ready for modeling")

        return self

    def get_featured_data(self):
        """Return featured DataFrame"""
        return self.df.copy()

    def print_summary(self):
        """Print summary"""
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING COMPLETE")
        print("=" * 70)
        print(f"Final shape: {self.df.shape}")
        print(f"Total features: {self.df.shape[1] - 2}")  # -2 for timestamp and target

        # Show all feature names
        feature_cols = [col for col in self.df.columns
                        if col not in ['timestamp', 'PKW', 'LKW', 'Buses', 'Total']]

        print(f"\nFeature columns ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i}. {col}")

        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    # Load clean data
    input_file = Path(config.PROCESSED_DATA_DIR) / "traffic_2023_01_clean.csv"

    if not input_file.exists():
        print(f" File not found: {input_file}")
        print("Please run: python src/data/preprocessing.py first")
    else:
        df = pd.read_csv(input_file)

        # Engineer features
        engineer = FeatureEngineer(df)
        df_featured = (engineer
                       .create_temporal_features()
                       .create_lag_features()
                       .create_rolling_features()
                       .drop_na_rows()
                       .get_featured_data())

        engineer.print_summary()

        # Save
        output = Path(config.PROCESSED_DATA_DIR) / "traffic_2023_01_featured.csv"
        df_featured.to_csv(output, index=False)
        print(f"✅ Saved to: {output}\n")
