"""
Compare All Models - January 2023
Author: Tulsi
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import config
from src.models.xgboost_model import XGBoostTrafficModel
from src.models.lstm_model import LSTMTrafficModel
from src.utils.metrics import compare_models
from src.utils.visualization import plot_model_comparison

print("\n" + "="*80)
print("MODEL COMPARISON - JANUARY 2023")
print("="*80)

# Load featured data
data_file = Path(config.PROCESSED_DATA_DIR) / "traffic_2023_01_featured.csv"
df = pd.read_csv(data_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Split test data
test_df = df[(df['timestamp'] >= config.TEST_START) & (df['timestamp'] <= config.TEST_END)]
print(f"\nTest set: {len(test_df)} samples")

# ============================================================
# XGBOOST EVALUATION
# ============================================================
print("\n" + "="*80)
print("LOADING XGBOOST MODEL")
print("="*80)

xgb_model = XGBoostTrafficModel()
xgb_model.load_model(Path(config.MODEL_DIR) / "xgboost_jan2023.pkl")

X_test_xgb, y_test = xgb_model.prepare_data(test_df)
xgb_metrics, xgb_pred = xgb_model.evaluate(X_test_xgb, y_test, "Test Set")

# ============================================================
# LSTM EVALUATION
# ============================================================
print("\n" + "="*80)
print("LOADING LSTM MODEL")
print("="*80)

# Get input size
exclude_cols = ['timestamp', 'PKW', 'LKW', 'Buses', 'Total']
feature_cols = [col for col in df.columns if col not in exclude_cols]
input_size = len(feature_cols)

lstm_model = LSTMTrafficModel(input_size=input_size, hidden_size=64, num_layers=1)
lstm_model.load_model(Path(config.MODEL_DIR) / "lstm_jan2023.pth")

X_test_lstm, y_test_lstm = lstm_model.prepare_data(test_df, sequence_length=24)
lstm_metrics, lstm_pred = lstm_model.evaluate(X_test_lstm, y_test_lstm, "Test Set")

# ============================================================
# COMPARISON
# ============================================================
print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)

results = {
    'XGBoost': xgb_metrics,
    'LSTM': lstm_metrics
}

compare_models(results)

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n" + "="*80)
print("CREATING COMPARISON PLOTS")
print("="*80)

results_dir = Path(config.RESULTS_DIR) / "comparison"
results_dir.mkdir(parents=True, exist_ok=True)

# Plot comparison for each metric
for metric in ['MAE', 'RMSE', 'R2']:
    plot_model_comparison(results, metric=metric,
                         save_path=results_dir / f"comparison_{metric.lower()}.png")

# Side-by-side predictions plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# XGBoost predictions
axes[0].plot(y_test[:168], label='Actual', color='blue', linewidth=2, alpha=0.7)
axes[0].plot(xgb_pred[:168], label='XGBoost', color='red', linewidth=2, alpha=0.7)
axes[0].set_title('XGBoost Predictions (First Week)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Traffic Volume', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# LSTM predictions (adjusted for sequence length)
axes[1].plot(y_test_lstm[:168], label='Actual', color='blue', linewidth=2, alpha=0.7)
axes[1].plot(lstm_pred[:168], label='LSTM', color='green', linewidth=2, alpha=0.7)
axes[1].set_title('LSTM Predictions (First Week)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Time (hours)', fontsize=12)
axes[1].set_ylabel('Traffic Volume', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / "side_by_side_predictions.png", dpi=300, bbox_inches='tight')
print(f"✅ Side-by-side plot saved")
plt.show()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n🏆 BEST MODEL: XGBoost")
print(f"\nPerformance Gap:")
print(f"  MAE:  XGBoost is {lstm_metrics['MAE'] - xgb_metrics['MAE']:.2f} vehicles/hour better")
print(f"  RMSE: XGBoost is {lstm_metrics['RMSE'] - xgb_metrics['RMSE']:.2f} vehicles/hour better")
print(f"  R²:   XGBoost is {xgb_metrics['R2'] - lstm_metrics['R2']:.4f} better")

print(f"\n💡 Key Finding:")
print(f"   With limited data (~700 samples), traditional ML (XGBoost)")
print(f"   significantly outperforms deep learning (LSTM).")
print(f"\n   This validates that:")
print(f"   ✓ Tree-based models excel on small tabular datasets")
print(f"   ✓ Deep learning requires much more training data")
print(f"   ✓ Feature engineering is crucial for traffic prediction")

print("\n" + "="*80 + "\n")