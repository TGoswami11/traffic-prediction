"""
Configuration - January 2023 Data
Author: Tulsi
THD Master Thesis
Purpose: Central configuration file for all project settings and paths
"""

import os  # For operating system operations (file paths, directories)

# Project paths - Define all directory locations
# Get absolute path of current file's directory (project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Main data folder
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Raw data folder (original BASt files)
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

# Processed data folder (cleaned, featured data)
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# External data folder (weather, events, etc.)
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')

# Models folder (trained XGBoost, LSTM models)
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# Results folder (plots, reports, comparisons)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Create directories if they don't exist
# exist_ok=True means no error if directory already exists
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# January 2023 Data Configuration
# Specify which month/year of data to use
DATA_YEAR = 2023  # Year of traffic data
DATA_MONTH = 1    # Month of traffic data (January)

# Train/Val/Test Split (70/15/15)
# Training set: 21 days (70% of 31 days)
TRAIN_START = "2023-01-01"  # Training starts from January 1st
TRAIN_END = "2023-01-21"    # Training ends on January 21st

# Validation set: 5 days (15% of 31 days)
VAL_START = "2023-01-22"  # Validation starts from January 22nd
VAL_END = "2023-01-26"    # Validation ends on January 26th

# Test set: 5 days (15% of 31 days)
TEST_START = "2023-01-27"  # Test starts from January 27th
TEST_END = "2023-01-31"    # Test ends on January 31st (last day)

# Model Parameters

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'max_depth': 6,           # Maximum tree depth (controls complexity)
    'learning_rate': 0.1,     # Step size for weight updates
    'n_estimators': 100,      # Number of trees in ensemble
    'random_state': 42        # Seed for reproducibility
}

# LSTM hyperparameters
LSTM_PARAMS = {
    'hidden_size': 64,        # Number of LSTM neurons per layer
    'num_layers': 1,          # Number of LSTM layers (1 = simple)
    'learning_rate': 0.001,   # Learning rate for Adam optimizer
    'batch_size': 32,         # Number of samples per training batch
    'epochs': 50              # Number of complete passes through training data
}

# Global random seed for reproducibility
RANDOM_SEED = 42  # Ensures same results every run

<<<<<<< Updated upstream
=======
# Confirmation message
>>>>>>> Stashed changes
print(f" Configuration loaded - January 2023")
