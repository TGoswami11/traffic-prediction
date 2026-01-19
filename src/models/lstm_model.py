"""
LSTM Model for Traffic Prediction
Author: Tulsi
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
##python train_lstm.pyfrom tqdm import tqdmpip install tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
import config
from src.utils.metrics import evaluate_model
from src.utils.visualization import plot_predictions, plot_scatter, plot_training_history


class LSTMModel(nn.Module):
    """LSTM Neural Network"""

    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)

        # Take output from last time step
        last_output = lstm_out[:, -1, :]

        # Fully connected layer
        output = self.fc(last_output)

        return output


class LSTMTrafficModel:
    """LSTM model wrapper for traffic prediction"""

    def __init__(self, input_size, hidden_size=64, num_layers=1,
                 dropout=0.2, learning_rate=0.001, device=None):
        """
        Initialize LSTM model

        Args:
            input_size: Number of features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
            device: torch device (cuda/cpu)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.history = {'train_loss': [], 'val_loss': []}
        self.feature_names = None

    def prepare_sequences(self, X, y, sequence_length=24):
        """
        Prepare sequential data for LSTM

        Args:
            X: Features (2D array)
            y: Target (1D array)
            sequence_length: Length of input sequence

        Returns:
            X_seq, y_seq: Sequential data
        """
        X_seq, y_seq = [], []

        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def prepare_data(self, df, sequence_length=24):
        """
        Prepare data from DataFrame

        Args:
            df: Featured DataFrame
            sequence_length: Sequence length for LSTM

        Returns:
            X_seq, y: Sequential features and target
        """
        # Remove non-feature columns
        exclude_cols = ['timestamp', 'PKW', 'LKW', 'Buses', 'Total']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].values
        y = df['PKW'].values

        self.feature_names = feature_cols

        # Create sequences
        X_seq, y_seq = self.prepare_sequences(X, y, sequence_length)

        return X_seq, y_seq

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=50, batch_size=32, patience=10):
        """
        Train the LSTM model

        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            patience: Early stopping patience
        """
        print("\n" + "=" * 70)
        print("TRAINING LSTM MODEL")
        print("=" * 70)
        print(f"Training samples: {len(X_train)}")
        print(f"Sequence length: {X_train.shape[1]}")
        print(f"Features: {X_train.shape[2]}")
        if X_val is not None:
            print(f"Validation samples: {len(X_val)}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)

        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            num_batches = 0

            # Mini-batch training
            for i in range(0, len(X_train_t), batch_size):
                batch_X = X_train_t[i:i + batch_size]
                batch_y = y_train_t[i:i + batch_size]

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / num_batches
            self.history['train_loss'].append(avg_train_loss)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t)
                    val_loss = self.criterion(val_outputs, y_val_t).item()
                    self.history['val_loss'].append(val_loss)

                # Print progress
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}] - "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\n⚠️ Early stopping at epoch {epoch + 1}")
                        break
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}")

        print("\n Training complete!")

    def predict(self, X):
        """
        Make predictions

        Args:
            X: Sequential input data

        Returns:
            predictions: Numpy array of predictions
        """
        self.model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy().flatten()

    def evaluate(self, X, y, dataset_name="Test"):
        """
        Evaluate model performance

        Args:
            X: Sequential features
            y: True values
            dataset_name: Name of dataset

        Returns:
            metrics, predictions
        """
        y_pred = self.predict(X)
        metrics = evaluate_model(y, y_pred, f"LSTM - {dataset_name}")
        return metrics, y_pred

    def plot_results(self, y_true, y_pred, dataset_name="Test", save_dir=None):
        """Plot prediction results"""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Predictions plot
        save_path = save_dir / f"lstm_{dataset_name.lower()}_predictions.png" if save_dir else None
        plot_predictions(y_true, y_pred,
                         f"LSTM - {dataset_name} Set Predictions",
                         save_path=save_path,
                         show_first_n=168)

        # Scatter plot
        save_path = save_dir / f"lstm_{dataset_name.lower()}_scatter.png" if save_dir else None
        plot_scatter(y_true, y_pred,
                     f"LSTM - {dataset_name} Set: Actual vs Predicted",
                     save_path=save_path)

    def plot_training_history(self, save_path=None):
        """Plot training history"""
        plot_training_history(self.history, "LSTM Training History", save_path)

    def save_model(self, filepath):
        """Save trained model"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
        print(f" Model saved to: {filepath}")

    def load_model(self, filepath):
        """Load trained model"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.history = checkpoint['history']
        print(f" Model loaded from: {filepath}")


# Test
if __name__ == "__main__":
    print("LSTM model module loaded successfully!")
    print("Use train_lstm.py to train the model.")
