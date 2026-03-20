"""
Traffic Prediction - Main Script
Author: Tulsi
Thesis: Master Elektro- und Informationstechnik, THD

Purpose: This is the main testing script that verifies the project setup
         and generates sample traffic data for testing.
"""

# ========================================================
# SECTION 1: IMPORT STATEMENTS
# =========================================================

# Import the warnings module to control warning messages
import warnings

# Disable all warning messages to keep output clean
# This prevents unnecessary warnings from cluttering the console
warnings.filterwarnings('ignore')

# Import pandas library for data manipulation and analysis
# pandas is used to create and work with DataFrames (tables)
import pandas as pd

# Import numpy library for numerical operations and array handling
# numpy is essential for mathematical operations and random number generation
import numpy as np

# Import datetime classes for working with dates and times
# datetime is used to create timestamps for traffic data
from datetime import datetime, timedelta

# Import the project configuration file
# config.py contains all project paths and settings
import config


# ============================================================================
# SECTION 2: MAIN TEST FUNCTION
# ============================================================================

def test_project_setup():
    """
    Test that project is set up correctly

    Purpose: This function performs 5 tests to verify that:
             1. Configuration files are accessible
             2. All required libraries can be imported
             3. Sample traffic data can be generated
             4. Data can be saved to disk
             5. Basic statistics can be calculated

    Returns: None (prints results to console)
    """

    # Print header section with decorative lines
     # Print a line of 80 equal signs for visual separation
    print("=" * 80)

    # Print the main title of the test
    print("TRAFFIC PREDICTION PROJECT - SETUP TEST")

    # Print another separator line
    print("=" * 80)


    # Test 1: Configuration
<<<<<<< Updated upstream
    print("\n Test 1: Configuration")
=======
    # Purpose: Verify that config.py file is working and paths are accessible


    # Print a blank line for spacing
    print("\n Test 1: Configuration")

    # Print the project root directory path from config file
    # This shows where the main project folder is located
>>>>>>> Stashed changes
    print(f"   Project Root: {config.PROJECT_ROOT}")

    # Print the data directory path from config file
    # This is where raw data files are stored
    print(f"   Data Directory: {config.DATA_DIR}")

    # Print the models directory path from config file
    # This is where trained ML models will be saved
    print(f"   Models Directory: {config.MODEL_DIR}")


    # Test 2: Import modules
<<<<<<< Updated upstream
    print("\n Test 2: Import Core Libraries")
=======
    # Purpose: Check if all required Python libraries are installed correctly


    # Print test 2 header
    print("\n Test 2: Import Core Libraries")

    # Start a try-except block to catch any import errors
>>>>>>> Stashed changes
    try:
        # Try to import pandas (data manipulation library)
        import pandas as pd

        # Try to import numpy (numerical computing library)
        import numpy as np

        # Try to import matplotlib.pyplot (plotting library)
        import matplotlib.pyplot as plt

        # Try to import seaborn (statistical visualization library)
        import seaborn as sns

        # Try to import scikit-learn (machine learning library)
        import sklearn

        # Try to import XGBoost (gradient boosting library)
        import xgboost

        # Try to import PyTorch (deep learning library)
        import torch

        # Try to import statsmodels (statistical modeling library)
        import statsmodels

        # If all imports succeed, print success message
        print("   All core libraries imported successfully!")

    # Catch any ImportError exceptions (when a library is not installed)
    except ImportError as e:
<<<<<<< Updated upstream
        print(f"    Error importing: {e}")
        return

    # Test 3: Generate sample data
    print("\n Test 3: Generate Sample Traffic Data")
=======
        # Print error message showing which library failed to import
        print(f"    Error importing: {e}")

        # Exit the function early since libraries are missing
        return

    # Test 3: Generate sample data
    # Purpose: Create realistic synthetic traffic data for testing-
>>>>>>> Stashed changes

    # Print test 3 header
    print("\n Test 3: Generate Sample Traffic Data")

    # Set random seed to 42 for reproducibility
    # This ensures we get the same random numbers every time
    np.random.seed(42)

    # Create 1 year of hourly data

    # Create a starting date: January 1, 2023
    start_date = datetime(2023, 1, 1)

    # Generate a list of datetime objects for every hour in a year
    # 24 hours/day × 365 days = 8760 hours total
    # Each hour is created by adding i hours to start_date
    dates = [start_date + timedelta(hours=i) for i in range(24 * 365)]

    # Simulate realistic traffic patterns
    # ------------------------------------------------------------------------

    # Extract the hour (0-23) from each timestamp
    # This will be used to create hourly traffic patterns
    hours = np.array([d.hour for d in dates])

    # Extract the day of week (0=Monday, 6=Sunday) from each timestamp
    # This will be used to create weekend vs weekday patterns
    days = np.array([d.weekday() for d in dates])

    # Calculate base traffic with realistic patterns
    # ------------------------------------------------------------------------

    # Set baseline traffic volume to 100 vehicles per hour
    base_traffic = 100

    # Create hourly pattern using sine wave
    # Peaks at 6am and 6pm (rush hours)
    # Amplitude of 50 means ±50 vehicles from baseline
    # The (hours - 6) shift makes peak at 6am
    # 2*pi/24 creates a full cycle over 24 hours
    hourly_pattern = 50 * np.sin(2 * np.pi * (hours - 6) / 24)

    # Create weekend effect: reduce traffic by 30 vehicles on weekends
    # days >= 5 creates boolean array (True for Sat/Sun)
    # Multiply by -30 to reduce traffic on weekends
    weekend_effect = -30 * (days >= 5)

    # Add random noise with mean=0 and std=10
    # This simulates natural variation in traffic
    noise = np.random.normal(0, 10, len(dates))

    # Combine all patterns to create final traffic counts
    # traffic = base + hourly_pattern + weekend_effect + noise
    traffic = base_traffic + hourly_pattern + weekend_effect + noise

    # Ensure minimum traffic of 10 vehicles
    # np.maximum compares each value with 10 and takes the larger
    traffic = np.maximum(traffic, 10)

    # Create DataFrame to organize the data
    # ------------------------------------------------------------------------

    # Create a pandas DataFrame with multiple columns
    df = pd.DataFrame({
        # Column 1: timestamp (datetime objects)
        'timestamp': dates,

        # Column 2: traffic count converted to integers
        'traffic_count': traffic.astype(int),

        # Column 3: hour of day (0-23)
        'hour': hours,

        # Column 4: day of week (0-6)
        'day_of_week': days,

        # Column 5: binary indicator (1=weekend, 0=weekday)
        'is_weekend': (days >= 5).astype(int)
    })

    # Print summary statistics about generated data
    # ------------------------------------------------------------------------

    # Print total number of rows (hours) generated
    print(f"   Generated {len(df)} hours of traffic data")

    # Print the date range: first and last timestamp
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Print the traffic range: minimum and maximum values
    print(f"   Traffic range: {df['traffic_count'].min()} to {df['traffic_count'].max()}")

    # ------------------------------------------------------------------------
    # Test 4: Save sample data
<<<<<<< Updated upstream
    print("\n Test 4: Save Sample Data")
=======
    # Purpose: Verify that data can be written to disk
    # ------------------------------------------------------------------------

    # Print test 4 header
    print("\n Test 4: Save Sample Data")

    # Create output file path by combining directory and filename
>>>>>>> Stashed changes
    output_path = config.PROCESSED_DATA_DIR + '/sample_traffic_data.csv'

    # Save DataFrame to CSV file
    # index=False means don't save row numbers as a column
    df.to_csv(output_path, index=False)

    # Print confirmation message with file path
    print(f"   Saved to: {output_path}")


    # Test 5: Basic statistics
<<<<<<< Updated upstream
    print("\n Test 5: Basic Statistics")
=======
    # Purpose: Calculate and display summary statistics


    # Print test 5 header
    print("\n Test 5: Basic Statistics")

    # Calculate and print mean (average) traffic count
    # .mean() calculates the arithmetic mean
    # :.2f formats the number to 2 decimal places
>>>>>>> Stashed changes
    print(f"   Mean traffic: {df['traffic_count'].mean():.2f}")

    # Calculate and print standard deviation of traffic
    # .std() measures the spread of the data
    print(f"   Std traffic: {df['traffic_count'].std():.2f}")

    # Calculate average traffic on weekdays only
    # df[df['is_weekend'] == 0] filters to weekdays only
    # Then calculate mean of those rows
    print(f"   Weekday avg: {df[df['is_weekend'] == 0]['traffic_count'].mean():.2f}")

    # Calculate average traffic on weekends only
    # df[df['is_weekend'] == 1] filters to weekends only
    print(f"   Weekend avg: {df[df['is_weekend'] == 1]['traffic_count'].mean():.2f}")

    # ----------------------------------
    # Print completion message and next steps
    # -------------------------------------

    # Print blank line for spacing
    print("\n" + "=" * 80)
<<<<<<< Updated upstream
    print(" PROJECT SETUP COMPLETE!")
=======

    # Print success message
    print(" PROJECT SETUP COMPLETE!")

    # Print separator
>>>>>>> Stashed changes
    print("=" * 80)

    # Print next steps instructions
    print("\nNext steps:")
    print("1. Start implementing data_loader.py")
    print("2. Then preprocessing.py")
    print("3. Then feature_engineering.py")
    print("4. Finally train your first model!")
    print("\n" + "=" * 80)


# ============================================================================
# SECTION 3: MAIN EXECUTION BLOCK
# ============================================================================

# Check if this script is being run directly (not imported)
# __name__ == "__main__" is True only when script is run directly
if __name__ == "__main__":
    # Call the test function to run all tests
    test_project_setup()