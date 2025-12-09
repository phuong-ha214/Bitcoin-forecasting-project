# Bitcoin-forecasting-project
📌 Project Overview

This project builds a time-series forecasting system for Bitcoin daily log-returns using:

ARIMA (traditional statistical model)

XGBoost (tree-based ML model)

LSTM (deep learning sequence model)

Stacking meta-model (Ridge Regression)

The workflow produces out-of-sample predictions for each base model, evaluates performance, and combines them using stacking for improved accuracy.

🚀 1. Setup Instructions
Clone the repository
git clone https://github.com/phuong-ha214/Bitcoin-forecasting-project.git
cd Bitcoin-forecasting-project

📦 Create the environment

Install required packages:

pip install -r requirements.txt


If you are using Google Colab, simply upload or mount your project folder and run the notebooks.

📊 2. Dataset

The raw dataset is stored in:

data/raw/btc_1d_data_2018_to_2025.csv


It contains:

Open/High/Low/Close prices

Volume

Timestamps

You may replace this file with updated data as long as the column structure remains the same.

🧹 3. Preprocessing & Feature Engineering
Run the preprocessing notebook:
notebooks/1_preprocessing.ipynb


This notebook:

✔ Loads raw BTC data
✔ Sorts by time
✔ Creates log-returns
✔ Generates lag features
✔ Normalizes features (fit on train only)
✔ Splits data into train/test
✔ Saves processed files into:

data/processed/btc_train.csv
data/processed/btc_test.csv

🤖 4. Train Base Models
A. ARIMA

Run:

notebooks/2_train_arima.ipynb


This notebook:

✔ Fits an ARIMA model on training returns
✔ Uses auto_arima to select optimal order
✔ Produces out-of-sample predictions aligned with test set
✔ Saves predictions to:

results/arima_preds_train.csv
results/arima_preds_test.csv

B. XGBoost

Run:

notebooks/3_train_xgboost.ipynb


This notebook:

✔ Loads train/test engineered features
✔ Trains XGBoost regressor
✔ Uses TimeSeriesSplit cross-validation
✔ Outputs predictions to:

results/xgb_preds_train.csv
results/xgb_preds_test.csv

C. LSTM

Run:

notebooks/4_train_lstm.ipynb


This notebook:

✔ Builds LSTM with Keras
✔ Uses 5-fold TimeSeriesSplit to generate out-of-sample predictions for stacking
✔ Produces final test predictions
✔ Saves results:

results/lstm_preds_train.csv
results/lstm_preds_test.csv

🧠 5. Train the Stacking Meta-Model

Run:

notebooks/5_train_stacking.ipynb


This notebook:

✔ Loads the 3 base model predictions
✔ Concatenates them into a feature matrix
✔ Fits Ridge Regression meta-model
✔ Generates final ensemble predictions
✔ Evaluates test performance
✔ Saves stacked outputs:

results/stacked_preds_test.csv
results/metrics_summary.csv

📈 6. Visualizations

Run:

notebooks/6_plot_results.ipynb


It will generate:

True vs Predicted returns

Rolling error curves

Model comparison table

Stacking improvement plots

Saved into:

results/plots/

📝 7. Reproducing Results (Step-by-Step Summary)
Run these notebooks in order:

✔ 1_preprocessing.ipynb

✔ 2_train_arima.ipynb

✔ 3_train_xgboost.ipynb

✔ 4_train_lstm.ipynb

✔ 5_train_stacking.ipynb

✔ 6_plot_results.ipynb

After running all, you will obtain:

ARIMA / XGBoost / LSTM model performance

Final stacked model performance

Visualizations + evaluation metrics

Final prediction CSV files

🧩 Notes

All models use training data only to avoid leakage.

All cross-validated predictions are out-of-sample to ensure fair stacking.

You can adjust hyperparameters in each notebook.
