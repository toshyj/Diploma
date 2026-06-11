"""
generate_table35.py — розрахунок значень для Таблиці 3.5.
Запускати: py generate_table35.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd

from ml.data_processor import DataProcessor
from ml.lstm_model import LSTMModel

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "household_power_consumption.txt")

# Завантаження даних і моделі
dp = DataProcessor(filepath=DATA_PATH, window_size=24, overlap_ratio=0.96)
dp.load().resample().clean().normalize()
_, _, X_test, _, _, y_test, ts_test = dp.make_windows()
y_pred_kw = dp.inverse_transform_power(
    LSTMModel(window_size=24, n_features=8).load(
        os.path.join(BASE_DIR, "models", "lstm_best.keras")
    ).predict(X_test)
)

# Тарифи
PEAK_TARIFF  = 4.32
NIGHT_TARIFF = 2.16
hours_arr    = np.arange(24)
tariffs      = np.where((hours_arr >= 7) & (hours_arr < 23), PEAK_TARIFF, NIGHT_TARIFF)

# Добовий профіль
profile = np.zeros(24)
counts  = np.zeros(24)
for i, ts in enumerate(ts_test):
    h = pd.Timestamp(ts).hour
    profile[h] += y_pred_kw[i]
    counts[h]  += 1
counts[counts == 0] = 1
profile /= counts

# Споживання в пікові та нічні години
peak_mask  = (hours_arr >= 7)  & (hours_arr < 23)
night_mask = (hours_arr >= 23) | (hours_arr < 7)
total      = profile.sum()
peak_pct   = profile[peak_mask].sum()  / total * 100
night_pct  = profile[night_mask].sum() / total * 100
cost_day   = float(np.sum(profile * tariffs))
cost_month = cost_day * 30

# Після оптимізації: 25% пікового → нічний
optimized = profile.copy()
pm = (hours_arr >= 18) & (hours_arr < 22)
nm = (hours_arr >= 23) | (hours_arr < 6)
shift = optimized[pm] * 0.25
optimized[pm] -= shift
optimized[nm] += shift.sum() / nm.sum()

total_opt      = optimized.sum()
peak_pct_opt   = optimized[peak_mask].sum()  / total_opt * 100
night_pct_opt  = optimized[night_mask].sum() / total_opt * 100
cost_day_opt   = float(np.sum(optimized * tariffs))
cost_month_opt = cost_day_opt * 30
saving_pct     = (cost_month - cost_month_opt) / cost_month * 100

print("=" * 55)
print("ТАБЛИЦЯ 3.5 — Економічна ефективність")
print("=" * 55)
print(f"{'Показник':<35} {'До':>8} {'Після':>8}")
print("-" * 55)
print(f"{'Споживання в пікові години':<35} {peak_pct:>7.0f}% {peak_pct_opt:>7.0f}%")
print(f"{'Споживання в нічні години':<35} {night_pct:>7.0f}% {night_pct_opt:>7.0f}%")
print(f"{'Вартість на добу (грн)':<35} {cost_day:>8.2f} {cost_day_opt:>8.2f}")
print(f"{'Орієнтовна вартість (міс., грн)':<35} {cost_month:>8.0f} {cost_month_opt:>8.0f}")
print(f"{'Економія (%)':<35} {'—':>8} {saving_pct:>7.1f}%")
print("=" * 55)
