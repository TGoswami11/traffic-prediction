"""
Train LSTM Model - January 2023
Author: Tulsi
Purpose: Train LSTM neural network for traffic prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config
from src.models.lstm_model import LSTMTrafficModel

print("\n" + "="*80)
print("LSTM MODEL TRAINING - JANUARY 2023")
print("="*80)

# LSTM hyperparameters
SEQUENCE_LENGTH = 24  # Use last 24 hours to predict next hour
HIDDEN_SIZE = 64  # Number of neurons in LSTM layer
NUM_LAYERS = 1  # Single LSTM layer
DROPOUT = 0.2  # Dropout rate for regularization (prevents overfitting)
LEARNING_RATE = 0.001  # Adam optimizer learning rate
BATCH_SIZE = 32  # Samples per training batch
EPOCHS = 50  # Maximum training iterations

# Load featured data with all engineered features
print("\nLoading data...")
data_file = Path(config.PROCESSED_DATA_DIR) / "traffic_2023_01_featured.csv"
df = pd.read_csv(data_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f" Loaded {len(df)} records")

# Split by date ranges from config
train_df = df[(df['timestamp'] >= config.TRAIN_START) & (df['timestamp'] <= config.TRAIN_END)]
val_df = df[(df['timestamp'] >= config.VAL_START) & (df['timestamp'] <= config.VAL_END)]
test_df = df[(df['timestamp'] >= config.TEST_START) & (df['timestamp'] <= config.TEST_END)]

print(f"\nData split:")
print(f"  Training:   {len(train_df)} samples")
print(f"  Validation: {len(val_df)} samples")
print(f"  Test:       {len(test_df)} samples")

# Calculate input size (number of features)
exclude_cols = ['timestamp', 'PKW', 'LKW', 'Buses', 'Total']  # Not features
feature_cols = [col for col in df.columns if col not in exclude_cols]
input_size = len(feature_cols)  # Number of input features for LSTM

print(f"\nInput features: {input_size}")

# Initialize LSTM model with architecture parameters
model = LSTMTrafficModel(
    input_size=input_size,  # Number of input features
    hidden_size=HIDDEN_SIZE,  # LSTM neurons
    num_layers=NUM_LAYERS,  # LSTM layers
    dropout=DROPOUT,  # Dropout rate
    learning_rate=LEARNING_RATE  # Optimizer learning rate
)

# Prepare sequential data (LSTM needs sequences, not single rows)
print(f"\nPreparing sequences (length={SEQUENCE_LENGTH})...")

# Create sequences: [t-24, t-23, ..., t-1] -> [t]
X_train, y_train = model.prepare_data(train_df, sequence_length=SEQUENCE_LENGTH)
X_val, y_val = model.prepare_data(val_df, sequence_length=SEQUENCE_LENGTH)
X_test, y_test = model.prepare_data(test_df, sequence_length=SEQUENCE_LENGTH)

# Print tensor shapes (samples, sequence_length, features)
print(f"Training sequences: {X_train.shape}")
print(f"Validation sequences: {X_val.shape}")
print(f"Test sequences: {X_test.shape}")

# Train model with early stopping (patience=10 epochs)
model.train(
    X_train, y_train,  # Training data
    X_val, y_val,  # Validation data for monitoring
    epochs=EPOCHS,  # Maximum 50 epochs
    batch_size=BATCH_SIZE,  # 32 samples per batch
    patience=10  # Stop if no improvement for 10 epochs
)

# Create output directory and save training curves
results_dir = Path(config.RESULTS_DIR) / "lstm"
results_dir.mkdir(parents=True, exist_ok=True)
model.plot_training_history(save_path=results_dir / "training_history.png")

# Evaluate on all datasets
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

# Calculate metrics (MAE, RMSE, R2, MAPE) for each set
train_metrics, train_pred = model.evaluate(X_train, y_train, "Training")
val_metrics, val_pred = model.evaluate(X_val, y_val, "Validation")
test_metrics, test_pred = model.evaluate(X_test, y_test, "Test")

# Plot predictions vs actual for test set
model.plot_results(y_test, test_pred, "Test", save_dir=results_dir)

# Save trained model to disk
model_path = Path(config.MODEL_DIR) / "lstm_jan2023.pth"
model.save_model(model_path)

# Print summary
print("\n" + "="*80)
print(" LSTM TRAINING COMPLETE!")
print("="*80)
print(f"Test MAE:  {test_metrics['MAE']:.2f}")  # Mean Absolute Error
print(f"Test RMSE: {test_metrics['RMSE']:.2f}")  # Root Mean Square Error
print(f"Test R²:   {test_metrics['R2']:.4f}")  # R-squared score
print("\nResults saved to:", results_dir)
print("="*80 + "\n")
