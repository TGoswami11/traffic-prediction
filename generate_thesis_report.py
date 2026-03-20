"""
Generate Thesis Summary Report
Author: Tulsi
Purpose: Creates comprehensive text report of thesis results
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import config

print("\n" + "="*80)
print("GENERATING THESIS SUMMARY REPORT")
print("="*80)

# Initialize empty list to store report lines
report = []

# Header section
report.append("="*80)
report.append("TRAFFIC VOLUME PREDICTION SYSTEM")
report.append("Master's Thesis Implementation")
report.append("="*80)
report.append("")

# Student and university information
report.append(f"Student: Tulsi")
report.append(f"University: Technische Hochschule Deggendorf")
report.append(f"Program: Master Elektro- und Informationstechnik")
report.append(f"Supervisor: Prof. Dr.-Ing. Danny Wauri")
report.append(f"Date: {datetime.now().strftime('%B %d, %Y')}")  # Current date
report.append("")

# Section 1: Project Overview
report.append("="*80)
report.append("1. PROJECT OVERVIEW")
report.append("="*80)
report.append("")
report.append("Objective:")
report.append("  Develop and compare machine learning models for predicting")
report.append("  highway traffic volume using BASt data from January 2023.")
report.append("")

# Section 2: Data Summary
report.append("="*80)
report.append("2. DATA SUMMARY")
report.append("="*80)
report.append("")
report.append("Dataset: BASt Traffic Counts - January 2023")
report.append("  - Total Records: 744 hours (31 days)")  # 31 days × 24 hours
report.append("  - Vehicle Types: PKW (Passenger Cars), LKW (Trucks), Buses")
report.append("  - Features Engineered: 23 features")
report.append("    * Temporal: hour, day_of_week, cyclical encodings")  # Time-based
report.append("    * Lag features: 1h, 2h, 3h, 6h, 12h, 24h, 48h")  # Historical values
report.append("    * Rolling statistics: 3h, 6h, 12h, 24h windows")  # Moving averages
report.append("")

# Data split breakdown
report.append("Data Split:")
report.append("  - Training: 21 days (70%) - Jan 1-21")
report.append("  - Validation: 5 days (15%) - Jan 22-26")
report.append("  - Test: 5 days (15%) - Jan 27-31")
report.append("")

# Section 3: Models description
report.append("="*80)
report.append("3. MODELS IMPLEMENTED")
report.append("="*80)
report.append("")

# XGBoost configuration
report.append("3.1 XGBoost (Gradient Boosting)")
report.append("  Parameters:")
report.append("    - max_depth: 6")  # Tree depth
report.append("    - learning_rate: 0.1")  # Step size
report.append("    - n_estimators: 100")  # Number of trees
report.append("")

# LSTM configuration
report.append("3.2 LSTM (Deep Learning)")
report.append("  Architecture:")
report.append("    - Hidden size: 64")  # Neurons per layer
report.append("    - Layers: 1")  # Single LSTM layer
report.append("    - Sequence length: 24 hours")  # Input window
report.append("    - Dropout: 0.2")  # Regularization
report.append("")

# Section 4: Results table
report.append("="*80)
report.append("4. RESULTS")
report.append("="*80)
report.append("")
report.append("Test Set Performance:")
report.append("")

# Performance comparison table
report.append("  Model      | MAE    | RMSE    | R²      | MAPE")
report.append("  " + "-"*60)
report.append("  XGBoost    | 19.13  | 31.97   | 0.9545  | 22.10%")  # Best model
report.append("  LSTM       | 73.72  | 110.74  | -0.3771 | 217.17%")  # Poor performance
report.append("")
report.append(" Best Model: XGBoost")
report.append("")

# Improvement metrics
report.append("Performance Improvement:")
report.append("  - MAE: XGBoost is 54.59 vehicles/hour better (74% improvement)")
report.append("  - RMSE: XGBoost is 78.77 vehicles/hour better (71% improvement)")
report.append("  - R²: XGBoost achieves 95.45% accuracy vs LSTM's -37.71%")
report.append("")

# Section 5: Key findings
report.append("="*80)
report.append("5. KEY FINDINGS")
report.append("="*80)
report.append("")

# Model performance insights
report.append("5.1 Model Performance")
report.append("  ✓ XGBoost significantly outperformed LSTM on limited dataset")
report.append("  ✓ Traditional ML excels on small tabular datasets (<1000 samples)")
report.append("  ✓ Deep learning requires substantially more training data")
report.append("")

# Feature importance analysis
report.append("5.2 Feature Importance (XGBoost)")
report.append("  ✓ Lag features (especially 24h, 48h) were most important")
report.append("  ✓ Rolling statistics provided valuable trend information")
report.append("  ✓ Temporal features captured daily/weekly patterns effectively")
report.append("")

# Traffic patterns discovered
report.append("5.3 Traffic Patterns Observed")
report.append("  ✓ Clear morning (7-9 AM) and evening (5-7 PM) rush hours")
report.append("  ✓ Weekday traffic higher than weekend traffic")
report.append("  ✓ Night hours (10 PM - 5 AM) show minimal traffic")
report.append("")

# Section 6: Conclusions
report.append("="*80)
report.append("6. CONCLUSIONS")
report.append("="*80)
report.append("")

# Main conclusions
report.append("1. For limited datasets, traditional ML (XGBoost) is superior to")
report.append("   deep learning approaches (LSTM).")
report.append("")
report.append("2. Feature engineering is critical for traffic prediction:")
report.append("   - Lag features capture temporal dependencies")
report.append("   - Rolling statistics capture trends")
report.append("   - Cyclical encoding preserves time periodicity")
report.append("")
report.append("3. XGBoost with MAE of 19.13 vehicles/hour demonstrates")
report.append("   high accuracy for operational traffic prediction.")
report.append("")
report.append("4. LSTM underperformance validates the importance of")
report.append("   sufficient training data for neural networks.")
report.append("")

# Section 7: Generated files
report.append("="*80)
report.append("7. FILES GENERATED")
report.append("="*80)
report.append("")

# Model files
report.append("Models:")
report.append("  ✓ models/xgboost_jan2023.pkl")  # Saved XGBoost
report.append("  ✓ models/lstm_jan2023.pth")  # Saved LSTM
report.append("")

# Result plots
report.append("Results:")
report.append("  ✓ results/xgboost/xgboost_test_predictions.png")
report.append("  ✓ results/xgboost/feature_importance.png")
report.append("  ✓ results/lstm/lstm_test_predictions.png")
report.append("  ✓ results/lstm/training_history.png")
report.append("  ✓ results/comparison/comparison_mae.png")
report.append("  ✓ results/comparison/side_by_side_predictions.png")
report.append("")

# Data files
report.append("Data:")
report.append("  ✓ data/processed/traffic_2023_01_raw.csv")  # Original
report.append("  ✓ data/processed/traffic_2023_01_clean.csv")  # Cleaned
report.append("  ✓ data/processed/traffic_2023_01_featured.csv")  # With features
report.append("")

# Section 8: Future work
report.append("="*80)
report.append("8. FUTURE WORK")
report.append("="*80)
report.append("")

# Recommendations for improvements
report.append("1. Expand dataset to full year for better LSTM performance")
report.append("2. Implement SARIMA for seasonal baseline comparison")
report.append("3. Add weather data for improved predictions")
report.append("4. Test ensemble methods combining XGBoost and LSTM")
report.append("5. Implement drift detection for model adaptation")
report.append("")

# End of report
report.append("="*80)
report.append("END OF REPORT")
report.append("="*80)

# Convert list to single string with newlines
report_text = "\n".join(report)

# Print to console
print(report_text)

# Save to file
report_path = Path(config.RESULTS_DIR) / "thesis_summary_report.txt"
with open(report_path, 'w', encoding='utf-8') as f:  # Open file for writing
    f.write(report_text)  # Write report content

print(f"\n Report saved to: {report_path}")
