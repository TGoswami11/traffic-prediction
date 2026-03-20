"""
Train XGBoost Model - January 2023
Author: Tulsi
Purpose: Train XGBoost model for traffic prediction
"""

# Suppress warning messages
import warnings
warnings.filterwarnings('ignore')

# Use non-interactive backend for plotting (no display needed)
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
from pathlib import Path
import config
from src.models.xgboost_model import XGBoostTrafficModel

print("\n" + "="*80)
print("XGBOOST MODEL TRAINING - JANUARY 2023")
print("="*80)

# Load featured data (with all engineered features)
print("\nLoading data...")
data_file = Path(config.PROCESSED_DATA_DIR) / "traffic_2023_01_featured.csv"
df = pd.read_csv(data_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f" Loaded {len(df)} records")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Split by date ranges (chronological split for time series)
train_df = df[(df['timestamp'] >= config.TRAIN_START) & (df['timestamp'] <= config.TRAIN_END)]
val_df = df[(df['timestamp'] >= config.VAL_START) & (df['timestamp'] <= config.VAL_END)]
test_df = df[(df['timestamp'] >= config.TEST_START) & (df['timestamp'] <= config.TEST_END)]

print(f"\nData split:")
print(f"  Training:   {len(train_df)} samples ({config.TRAIN_START} to {config.TRAIN_END})")
print(f"  Validation: {len(val_df)} samples ({config.VAL_START} to {config.VAL_END})")
print(f"  Test:       {len(test_df)} samples ({config.TEST_START} to {config.TEST_END})")

# Initialize XGBoost model with default parameters from config
model = XGBoostTrafficModel()

# Prepare X (features) and y (target) for each set
X_train, y_train = model.prepare_data(train_df)
X_val, y_val = model.prepare_data(val_df)
X_test, y_test = model.prepare_data(test_df)

# Train model with validation monitoring
model.train(X_train, y_train, X_val, y_val)

# Evaluate on all datasets
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

# Calculate metrics (MAE, RMSE, R2, MAPE) for each set
train_metrics, train_pred = model.evaluate(X_train, y_train, "Training")
val_metrics, val_pred = model.evaluate(X_val, y_val, "Validation")
test_metrics, test_pred = model.evaluate(X_test, y_test, "Test")

# Create visualizations
results_dir = Path(config.RESULTS_DIR) / "xgboost"
model.plot_results(y_test, test_pred, "Test", save_dir=results_dir)  # Predictions plot
model.plot_feature_importance(top_n=15, save_path=results_dir / "feature_importance.png")  # Top 15 features

# Save trained model to disk
model_path = Path(config.MODEL_DIR) / "xgboost_jan2023.pkl"
model.save_model(model_path)

# Print summary
print("\n" + "="*80)
print(" TRAINING COMPLETE!")
print("="*80)
print(f"Test MAE:  {test_metrics['MAE']:.2f}")  # Mean Absolute Error
print(f"Test RMSE: {test_metrics['RMSE']:.2f}")  # Root Mean Square Error
print(f"Test R²:   {test_metrics['R2']:.4f}")  # R-squared
print("\nResults saved to:", results_dir)
print("="*80 + "\n")
<<<<<<< Updated upstream
=======


print(X_train)
print(X_train.shape, y_train.shape)
>>>>>>> Stashed changes
