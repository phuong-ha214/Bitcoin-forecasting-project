# Bitcoin Time-Series Forecasting Project

## 📌 Project Overview
This project builds a time-series forecasting system for Bitcoin daily log-returns using:

- **ARIMA** (statistical model)
- **XGBoost** (tree-based ML model)
- **LSTM** (deep learning sequence model)
- **Stacking meta-model** (Ridge Regression)

The workflow produces out-of-sample predictions for each base model and combines them to improve accuracy.

---

# 🚀 1. Setup Instructions

### Clone the repository
```bash
git clone https://github.com/phuong-ha214/Bitcoin-forecasting-project.git
cd Bitcoin-forecasting-project
```

### Install requirements
```bash
pip install -r requirements.txt
```

---

# 📊 2. Dataset

Raw dataset is stored in:

```
data/raw/btc_1d_data_2018_to_2025.csv
```

Contains daily:

- OHLC (open, high, low, close) prices  
- Volume  
- Timestamp
- Quote asset volume
- Number of trades
- Taker buy base asset volume
- Taker buy quote asset volume
- Ignore

---

# 🧹 3. Preprocessing & Feature Engineering

Run:

```
codes/bitcoin-preprocessing.ipynb
```

This notebook performs the **full end-to-end preprocessing pipeline** used across all models.

---

### ✔ Load & clean raw Bitcoin data
- Loads `btc_1d_data_2018_to_2025.csv`
- Parses timestamps into datetime
- Sorts by `"Open time"` and removes duplicates
- Computes daily log-returns

---

### ✔ Create target variable (`log_return_norm`)
- Computes `log_return = log(Close / Close.shift(1))`
- Runs **ADF stationarity test**  
- Computes rolling and EWM volatility  
- Creates **volatility-normalized log return**:

```
log_return_norm = log_return / ewm_vol_7
```

This becomes the final supervised learning target.

---

### ✔ Feature engineering

#### **Lagged returns**
- `lag_1`, `lag_2`, `lag_3`

#### **Volatility features**
- `vol_7`, `vol_14`, `vol_30`
- `ewm_vol_7`
- `vol_ratio = vol_7 / vol_30`

#### **Smoothed returns**
- `ma_return_3`
- `ma_return_5`

#### **Volume features**
- `volume_change`
- `volume_vol_5`

#### **Price-action & technical signals**
- `high_low_range`
- `close_open_ratio`
- `momentum_3`

All features are shifted appropriately to avoid look-ahead leakage.

---

### ✔ Train/Test split + Scaling (No leakage)

- Split dataset **80% train / 20% test**
- Fit `StandardScaler()` **only on training features**
- Apply scaling to full dataset and attach new columns:

```
<feature>_scaled
```

---

### ✔ LSTM sequence generation

Using a **sequence length of 7**:

```
X_train_lstm, y_train_lstm
X_test_lstm,  y_test_lstm
```

Sequences maintain chronological ordering to ensure **no temporal leakage**.

---

### ✔ Files saved to `preprocessed/`

```
preprocessed/btc_train.csv
preprocessed/btc_test.csv

preprocessed/X_train_lstm.npy
preprocessed/y_train_lstm.npy
preprocessed/X_test_lstm.npy
preprocessed/y_test_lstm.npy

preprocessed/scaler.pkl
preprocessed/feature_info.pkl
```

`feature_info.pkl` stores:
- list of base features  
- list of scaled features  
- LSTM sequence length  
- target variable name  

---

# 🤖 4. Train Base Models

## A. ARIMA

Run:
```
codes/arima.ipynb
```

This notebook:

- Fits Auto-ARIMA to `log_return_norm`
- Predicts on test set
- Plots **Actual vs Predicted**
- Saves predictions:

```
predictions/arima_train_preds.npy
predictions/arima_test_preds.npy
```

---

## B. XGBoost

Run:
```
codes/xgboost.ipynb
```

This notebook:

- Trains on engineered + scaled features  
- Uses TimeSeriesSplit  
- Plots **Actual vs Predicted**  
- Saves predictions:

```
predictions/xgb_train_preds.npy
predictions/xgb_test_preds.npy
```

---

## C. LSTM

Run:
```
codes/lstm.ipynb
```

This notebook:

- Loads LSTM sequences from preprocessing
- Trains an LSTM model with early stopping
- Plots **Actual vs Predicted**
- Saves predictions:

```
predictions/lstm_train_preds.npy
predictions/lstm_test_preds.npy
```

---

# 🧠 5. Stacking Meta-Model

Run:
```
codes/stacking.ipynb
```

This notebook:

- Loads ARIMA, XGBoost, and LSTM predictions  
- Builds stacking feature matrix  
- Fits **Ridge Regression** as the meta-learner  
- Plots stacked **Actual vs Predicted**
- Saves predictions:

```
predictions/meta_ridge_test_preds.npy
```

---

# 📈 6. Exploratory Data Analysis

Run:
```
codes/data-exploration.ipynb
```

This notebook performs EDA:

- Bitcoin daily closing price
- Bitcoin daily log returns
- Rolling volatility (7-day, 14-day, 30-day)
- Autocorrelation (ACF) of log returns
- Distribution of log returns (histogram + KDE)
- Correlation heatmap of numeric features
- Scatterplot: today's return vs tomorrow's return

It **does not save figures**, but is useful for understanding the dataset.

---

# 📝 7. Reproducing the Entire Pipeline

Run notebooks **in this order**:

1. `data-exploration.ipynb`
2. `bitcoin-preprocessing.ipynb`  
3. `arima.ipynb`  
4. `xgboost.ipynb`  
5. `lstm.ipynb`  
6. `stacking.ipynb`

---
