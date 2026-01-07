"""
Train LSTM Model - January 2023
Author: Tulsi
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config
from src.models.lstm_model import LSTMTrafficModel

print("\n" + "="*80)
print("LSTM MODEL TRAINING - JANUARY 2023")
print("="*80)

# Parameters
SEQUENCE_LENGTH = 24  # Use 24 hours to predict next hour
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50

# Load featured data
print("\nLoading data...")
data_file = Path(config.PROCESSED_DATA_DIR) / "traffic_2023_01_featured.csv"
df = pd.read_csv(data_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"✅ Loaded {len(df)} records")

# Split data by date
train_df = df[(df['timestamp'] >= config.TRAIN_START) & (df['timestamp'] <= config.TRAIN_END)]
val_df = df[(df['timestamp'] >= config.VAL_START) & (df['timestamp'] <= config.VAL_END)]
test_df = df[(df['timestamp'] >= config.TEST_START) & (df['timestamp'] <= config.TEST_END)]

print(f"\nData split:")
print(f"  Training:   {len(train_df)} samples")
print(f"  Validation: {len(val_df)} samples")
print(f"  Test:       {len(test_df)} samples")

# Initialize model (need input size first)
exclude_cols = ['timestamp', 'PKW', 'LKW', 'Buses', 'Total']
feature_cols = [col for col in df.columns if col not in exclude_cols]
input_size = len(feature_cols)

print(f"\nInput features: {input_size}")

model = LSTMTrafficModel(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    learning_rate=LEARNING_RATE
)

# Prepare sequential data
print(f"\nPreparing sequences (length={SEQUENCE_LENGTH})...")

X_train, y_train = model.prepare_data(train_df, sequence_length=SEQUENCE_LENGTH)
X_val, y_val = model.prepare_data(val_df, sequence_length=SEQUENCE_LENGTH)
X_test, y_test = model.prepare_data(test_df, sequence_length=SEQUENCE_LENGTH)

print(f"Training sequences: {X_train.shape}")
print(f"Validation sequences: {X_val.shape}")
print(f"Test sequences: {X_test.shape}")

# Train
model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    patience=10
)

# Plot training history
results_dir = Path(config.RESULTS_DIR) / "lstm"
results_dir.mkdir(parents=True, exist_ok=True)
model.plot_training_history(save_path=results_dir / "training_history.png")

# Evaluate on all sets
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

train_metrics, train_pred = model.evaluate(X_train, y_train, "Training")
val_metrics, val_pred = model.evaluate(X_val, y_val, "Validation")
test_metrics, test_pred = model.evaluate(X_test, y_test, "Test")

# Plot results
model.plot_results(y_test, test_pred, "Test", save_dir=results_dir)

# Save model
model_path = Path(config.MODEL_DIR) / "lstm_jan2023.pth"
model.save_model(model_path)

print("\n" + "="*80)
print("✅ LSTM TRAINING COMPLETE!")
print("="*80)
print(f"Test MAE:  {test_metrics['MAE']:.2f}")
print(f"Test RMSE: {test_metrics['RMSE']:.2f}")
print(f"Test R²:   {test_metrics['R2']:.4f}")
print("\nResults saved to:", results_dir)
print("="*80 + "\n")