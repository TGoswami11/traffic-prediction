def parse_bast_file(filepath):
    """
    Parse BASt traffic data CSV

    Args:
        filepath: Path to bast_2023_01_converted.csv

    Returns:
        DataFrame with columns: timestamp, PKW, LKW, Buses, Total
    """
    # Read CSV with proper encoding
    # Parse German date formats
    # Rename columns to English
    # Convert data types
    # Return clean DataFrame