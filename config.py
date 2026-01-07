"""
Configuration - January 2023 Data
Author: Tulsi
THD Master Thesis
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Create directories
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# January 2023 Data Configuration
DATA_YEAR = 2023
DATA_MONTH = 1

# Train/Val/Test Split (70/15/15)
TRAIN_START = "2023-01-01"
TRAIN_END = "2023-01-21"      # 21 days (70%)
VAL_START = "2023-01-22"
VAL_END = "2023-01-26"        # 5 days (15%)
TEST_START = "2023-01-27"
TEST_END = "2023-01-31"       # 5 days (15%)

# Model Parameters
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'random_state': 42
}

LSTM_PARAMS = {
    'hidden_size': 64,
    'num_layers': 1,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50
}

RANDOM_SEED = 42

print(f"✅ Configuration loaded - January 2023")