"""
Load Converted BaSt CSV Data - January 2023
Author: Tulsi
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


def load_january_csv():
    """Load the converted CSV file"""
    print("\n" + "=" * 70)
    print("LOADING BAST JANUARY 2023 DATA")
    print("=" * 70)

    # Look for CSV file
    csv_file = Path(config.RAW_DATA_DIR) / "bast_2023_01_converted.csv"

    if not csv_file.exists():
        print(f" File not found: {csv_file}")
        print(f"\nPlease copy 'bast_2023_01_converted.csv' to:")
        print(f"  {config.RAW_DATA_DIR}")
        raise FileNotFoundError("CSV file not found")

    # Load CSV
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"\n Data loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Duration: {(df['timestamp'].max() - df['timestamp'].min()).days + 1} days")

    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nStatistics:")
    print(df[['PKW', 'LKW', 'Buses']].describe())

    print("=" * 70 + "\n")

    return df


if __name__ == "__main__":
    df = load_january_csv()

    # Save to processed
    output = Path(config.PROCESSED_DATA_DIR) / "traffic_2023_01_raw.csv"
    df.to_csv(output, index=False)
    print(f" Saved to: {output}\n")
