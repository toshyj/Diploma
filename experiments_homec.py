"""
experiments_homec.py — навчання LSTM-моделі на датасеті HomeC.csv
Запускати: py experiments_homec.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ml.data_processor_homec import DataProcessorHomeC
from ml.lstm_model import LSTMModel

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "HomeC.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
#  Підготовка даних                                                   #
# ------------------------------------------------------------------ #
print("=" * 60)
print("Навчання моделі на датасеті HomeC.csv")
print("=" * 60)

dp = DataProcessorHomeC(filepath=DATA_PATH, window_size=24, overlap_ratio=0.96)
dp.load().resample().clean().normalize()
X_train, X_val, X_test, y_train, y_val, y_test, ts_test = dp.make_windows()

y_test_kw = dp.inverse_transform_power(y_test)

# ------------------------------------------------------------------ #
#  Навчання                                                           #
# ------------------------------------------------------------------ #
print("\nНавчання LSTM...")
lstm = LSTMModel(window_size=24, n_features=8, lstm_units=64, dropout_rate=0.2)
lstm.build()
lstm.train(
    X_train, y_train,
    X_val,   y_val,
    epochs=100,
    batch_size=32,
    patience=10,
    model_path=os.path.join(MODELS_DIR, "lstm_homec.keras"),
)

# ------------------------------------------------------------------ #
#  Оцінка                                                             #
# ------------------------------------------------------------------ #
y_pred_kw = dp.inverse_transform_power(lstm.predict(X_test))
metrics   = lstm.evaluate(y_test_kw, y_pred_kw)

print("\n=== Результати ===")
print(f"MAE  : {metrics['MAE']} кВт")
print(f"RMSE : {metrics['RMSE']} кВт")
print(f"MAPE : {metrics['MAPE']} %")
print(f"\nМодель збережено: models/lstm_homec.keras")
print("Готово!")
