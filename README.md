# Traffic Volume Prediction System

Master's Thesis Implementation  
**Student:** Tulsi  
**University:** Technische Hochschule Deggendorf  
**Program:** Master Elektro- und Informationstechnik  
**Supervisor:** Prof. Dr.-Ing. Danny Wauri

## Project Overview

Machine learning system for predicting highway traffic volumes using BASt data from January 2023.

## Dataset

- **Source:** BASt (German Federal Highway Research Institute)
- **Period:** January 2023 (744 hours)
- **Features:** 23 engineered features including temporal, lag, and rolling statistics

## Models Implemented

1. **XGBoost** (Winner! 🏆)
   - MAE: 19.13
   - R²: 0.9545 (95.45% accuracy)
   
2. **LSTM** (Deep Learning)
   - MAE: 73.72
   - R²: -0.3771

## Key Findings

✅ **XGBoost outperformed LSTM by 74%** on limited dataset  
✅ Traditional ML excels on small tabular datasets  
✅ Deep learning requires much more training data  
✅ Feature engineering is critical for traffic prediction

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Processing
```bash
python src/data/data_loader.py
python src/data/preprocessing.py
python src/data/feature_engineering.py
```

### 2. Model Training
```bash
python train_xgboost.py
python train_lstm.py
```

### 3. Model Comparison
```bash
python compare_models.py
```

### 4. Generate Report
```bash
python generate_thesis_report.py
```

## Results

**Test Set Performance:**

| Model   | MAE   | RMSE   | R²     | MAPE    |
|---------|-------|--------|--------|---------|
| XGBoost | 19.13 | 31.97  | 0.9545 | 22.10%  |
| LSTM    | 73.72 | 110.74 | -0.3771| 217.17% |

## Project Structure
```
traffic_prediction/
├── data/                  # Data files
├── models/                # Trained models
├── results/               # Plots and reports
├── src/                   # Source code
│   ├── data/             # Data processing
│   ├── models/           # Model implementations
│   └── utils/            # Utilities
├── notebooks/            # Jupyter notebooks
├── train_xgboost.py     # XGBoost training
├── train_lstm.py        # LSTM training
└── compare_models.py    # Model comparison
```

## License

Academic use only - THD Master's Thesistraffic_prediction/


│
├── venv/                           # Virtual environment (auto-created)
│
├── data/                           # Data folders
│   ├── raw/                        # Raw BASt data
│   ├── processed/                  # Cleaned data
│   └── external/                   # Weather, calendar
│
├── notebooks/                      # Jupyter notebooks (4 files)
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_drift_detection.ipynb
│
├── src/                           # Source code
│   ├── __init__.py                # File 5
│   │
│   ├── data/                      # Data processing (4 files)
│   │   ├── __init__.py            # File 6
│   │   ├── data_loader.py         # File 7
│   │   ├── preprocessing.py       # File 8
│   │   └── feature_engineering.py # File 9
│   │
│   ├── models/                    # ML models (6 files)
│   │   ├── __init__.py            # File 10
│   │   ├── sarima_model.py        # File 11
│   │   ├── xgboost_model.py       # File 12
│   │   ├── lstm_model.py          # File 13
│   │   ├── gru_model.py           # File 14
│   │   └── ensemble_model.py      # File 15
│   │
│   ├── drift/                     # Drift detection (3 files)
│   │   ├── __init__.py            # File 16
│   │   ├── drift_detector.py      # File 17
│   │   └── adaptation.py          # File 18
│   │
│   └── utils/                     # Utilities (3 files)
│       ├── __init__.py            # File 19
│       ├── metrics.py             # File 20
│       └── visualization.py       # File 21
│
├── models/                        # Saved trained models
├── results/                       # Outputs, plots
├── tests/                         # Unit tests
├── docs/                          # Documentation
│
├── main.py                        # File 1 - Main script
├── config.py                      # File 2 - Configuration
├── requirements.txt               # File 3 - Dependencies
└── README.md                      # File 4 - Project info

