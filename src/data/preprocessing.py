"""
Data Preprocessing - January 2023
Author: Tulsi
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


class TrafficPreprocessor:
    """Clean and preprocess traffic data"""

    def __init__(self, df):
        self.df = df.copy()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

    def handle_missing_values(self):
        """Fill missing values with interpolation"""
        print("\n" + "=" * 70)
        print("HANDLING MISSING VALUES")
        print("=" * 70)

        missing_before = self.df.isnull().sum().sum()
        print(f"Missing values before: {missing_before}")

        if missing_before > 0:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].interpolate(
                method='linear',
                limit=3
            )
            self.df = self.df.dropna()

            missing_after = self.df.isnull().sum().sum()
            print(f"Missing values after: {missing_after}")
            print(f"✅ Cleaned {missing_before - missing_after} missing values")
        else:
            print("✅ No missing values found")

        return self

    def remove_outliers(self, n_std=3):
        """Remove statistical outliers"""
        print("\n" + "=" * 70)
        print(f"REMOVING OUTLIERS (threshold = {n_std} std)")
        print("=" * 70)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        before_len = len(self.df)

        for col in numeric_cols:
            mean = self.df[col].mean()
            std = self.df[col].std()
            lower = mean - n_std * std
            upper = mean + n_std * std

            mask = (self.df[col] >= lower) & (self.df[col] <= upper)
            outliers = (~mask).sum()

            if outliers > 0:
                print(f"  {col}: removed {outliers} outliers")
                self.df = self.df[mask]

        after_len = len(self.df)
        total_removed = before_len - after_len

        print(f"\n✅ Total removed: {total_removed} rows ({total_removed / before_len * 100:.1f}%)")

        return self

    def ensure_hourly_continuity(self):
        """Ensure complete hourly data for January"""
        print("\n" + "=" * 70)
        print("ENSURING HOURLY CONTINUITY")
        print("=" * 70)

        # Create full January range
        start = pd.Timestamp('2023-01-01 00:00:00')
        end = pd.Timestamp('2023-01-31 23:00:00')
        full_range = pd.date_range(start=start, end=end, freq='H')

        print(f"Expected: {len(full_range)} hours (31 days × 24)")
        print(f"Actual: {len(self.df)} hours")

        # Reindex
        self.df = self.df.set_index('timestamp')
        self.df = self.df.reindex(full_range)
        self.df.index.name = 'timestamp'
        self.df = self.df.reset_index()

        # Fill any gaps
        missing = self.df.isnull().sum().sum()
        if missing > 0:
            print(f"Filling {missing} missing values from gaps...")
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].interpolate(method='linear')

        print(f"✅ Complete: {len(self.df)} hourly records")

        return self

    def get_cleaned_data(self):
        """Return cleaned DataFrame"""
        return self.df.copy()

    def print_summary(self):
        """Print summary"""
        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE")
        print("=" * 70)
        print(f"Final shape: {self.df.shape}")
        print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"Duration: {(self.df['timestamp'].max() - self.df['timestamp'].min()).days + 1} days")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    # Load raw data
    input_file = Path(config.PROCESSED_DATA_DIR) / "traffic_2023_01_raw.csv"

    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        print("Please run: python src/data/data_loader.py first")
    else:
        df = pd.read_csv(input_file)

        # Clean
        preprocessor = TrafficPreprocessor(df)
        df_clean = (preprocessor
                    .handle_missing_values()
                    .remove_outliers(n_std=3)
                    .ensure_hourly_continuity()
                    .get_cleaned_data())

        preprocessor.print_summary()

        # Save
        output = Path(config.PROCESSED_DATA_DIR) / "traffic_2023_01_clean.csv"
        df_clean.to_csv(output, index=False)
        print(f"✅ Saved to: {output}\n")