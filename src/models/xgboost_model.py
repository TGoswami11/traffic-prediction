"""
XGBoost Model for Traffic Prediction
Author: Tulsi
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
import config
from src.utils.metrics import evaluate_model
from src.utils.visualization import plot_predictions, plot_scatter, plot_feature_importance


class XGBoostTrafficModel:
    """XGBoost model for traffic prediction"""

    def __init__(self, params=None):
        """
        Initialize XGBoost model

        Args:
            params: XGBoost parameters (uses config.XGBOOST_PARAMS if None)
        """
        if params is None:
            params = config.XGBOOST_PARAMS

        self.params = params
        self.model = xgb.XGBRegressor(**params)
        self.feature_names = None

    def prepare_data(self, df):
        """
        Prepare data for training

        Args:
            df: Featured DataFrame with timestamp

        Returns:
            X, y: Features and target
        """
        # Remove non-feature columns
        exclude_cols = ['timestamp', 'PKW', 'LKW', 'Buses', 'Total']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].values
        y = df['PKW'].values

        self.feature_names = feature_cols

        return X, y

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        print("\n" + "=" * 70)
        print("TRAINING XGBOOST MODEL")
        print("=" * 70)
        print(f"Training samples: {len(X_train)}")
        if X_val is not None:
            print(f"Validation samples: {len(X_val)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"\nModel parameters:")
        for key, value in self.params.items():
            print(f"  {key}: {value}")

        # Train
        if X_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

        print("\n✅ Training complete!")

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def evaluate(self, X, y, dataset_name="Test"):
        """
        Evaluate model performance

        Args:
            X: Features
            y: True values
            dataset_name: Name of dataset (for printing)

        Returns:
            dict: Metrics dictionary
        """
        y_pred = self.predict(X)
        metrics = evaluate_model(y, y_pred, f"XGBoost - {dataset_name}")
        return metrics, y_pred

    def plot_results(self, y_true, y_pred, dataset_name="Test", save_dir=None):
        """
        Plot prediction results

        Args:
            y_true: True values
            y_pred: Predicted values
            dataset_name: Name of dataset
            save_dir: Directory to save plots
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Predictions plot
        save_path = save_dir / f"xgboost_{dataset_name.lower()}_predictions.png" if save_dir else None
        plot_predictions(y_true, y_pred,
                         f"XGBoost - {dataset_name} Set Predictions",
                         save_path=save_path,
                         show_first_n=168)  # Show 1 week

        # Scatter plot
        save_path = save_dir / f"xgboost_{dataset_name.lower()}_scatter.png" if save_dir else None
        plot_scatter(y_true, y_pred,
                     f"XGBoost - {dataset_name} Set: Actual vs Predicted",
                     save_path=save_path)

    def plot_feature_importance(self, top_n=15, save_path=None):
        """Plot feature importance"""
        importance = self.model.feature_importances_

        if self.feature_names:
            plot_feature_importance(
                self.feature_names,
                importance,
                title="XGBoost - Feature Importance",
                top_n=top_n,
                save_path=save_path
            )

    def save_model(self, filepath):
        """Save trained model"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"✅ Model saved to: {filepath}")

    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        print(f"✅ Model loaded from: {filepath}")


# Test/Demo
if __name__ == "__main__":
    print("XGBoost model module loaded successfully!")
    print("Use train_xgboost.py to train the model.")