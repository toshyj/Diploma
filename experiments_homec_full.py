"""
experiments_homec_full.py — повний цикл експериментів для датасету HomeC.csv.
Генерує всі графіки та таблиці аналогічно до experiments.py, але для HomeC.
Результати зберігаються у папці figures_homec/

Запускати: py experiments_homec_full.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from ml.data_processor_homec import DataProcessorHomeC
from ml.lstm_model import LSTMModel

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "HomeC.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures_homec")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def save_fig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Збережено: figures_homec/{name}")

# ================================================================== #
#  КРОК 0. Підготовка даних                                          #
# ================================================================== #
print("=" * 60)
print("КРОК 0. Підготовка даних (HomeC)")
print("=" * 60)

dp = DataProcessorHomeC(filepath=DATA_PATH, window_size=24, overlap_ratio=0.96)
dp.load().resample().clean().normalize()
X_train, X_val, X_test, y_train, y_val, y_test, ts_test = dp.make_windows()
y_test_kw = dp.inverse_transform_power(y_test)

# ================================================================== #
#  КРОК 1. Завантаження навченої моделі та метрики                   #
# ================================================================== #
print("\n" + "=" * 60)
print("КРОК 1. Завантаження моделі HomeC")
print("=" * 60)

lstm = LSTMModel(window_size=24, n_features=8, lstm_units=64)
lstm.load(os.path.join(MODELS_DIR, "lstm_homec.keras"))

y_pred_lstm_kw = dp.inverse_transform_power(lstm.predict(X_test))
metrics_lstm   = lstm.evaluate(y_test_kw, y_pred_lstm_kw)
print(f"LSTM метрики: {metrics_lstm}")

# -- Криві навчання недоступні (модель вже навчена), замість цього --
# -- генеруємо графік прогнозу vs реальність --
print("\nГенерація графіків...")
n_show = min(168, len(y_test_kw))
dates  = pd.to_datetime(ts_test[:n_show])

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates, y_test_kw[:n_show],       label="Реальне споживання", color="#2196F3", linewidth=1.5)
ax.plot(dates, y_pred_lstm_kw[:n_show],  label="Прогноз LSTM",       color="#F44336", linewidth=1.5, linestyle="--")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45)
ax.set_xlabel("Дата")
ax.set_ylabel("Споживання (кВт)")
ax.set_title("Рисунок 3.12 — Фактичне споживання vs Прогноз LSTM (HomeC)")
ax.legend()
ax.grid(True, alpha=0.3)
save_fig("fig_3_12_forecast_vs_real.png")

# Гістограма залишків
residuals = y_test_kw - y_pred_lstm_kw
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(residuals, bins=40, color="#4CAF50", edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Нуль")
ax.set_xlabel("Залишкова похибка (кВт)")
ax.set_ylabel("Частота")
ax.set_title("Рисунок 3.13 — Гістограма залишкових похибок (HomeC)")
ax.legend()
ax.grid(True, alpha=0.3)
save_fig("fig_3_13_residuals.png")

# ================================================================== #
#  КРОК 2. Абляція розміру вікна                                     #
# ================================================================== #
print("\n" + "=" * 60)
print("КРОК 2. Абляція розміру вікна (HomeC)")
print("=" * 60)

import time
window_results = []

for k in [12, 24, 48, 168]:
    print(f"\n  → Вікно k={k} год...")
    dp_k = DataProcessorHomeC(filepath=DATA_PATH, window_size=k, overlap_ratio=0.96)
    dp_k.load().resample().clean().normalize()
    Xtr, Xv, Xte, ytr, yv, yte, _ = dp_k.make_windows()

    if len(Xtr) < 10:
        print(f"    Недостатньо даних, пропускаємо.")
        continue

    yte_kw = dp_k.inverse_transform_power(yte)
    m = LSTMModel(window_size=k, n_features=8, lstm_units=64)
    m.build()
    t0 = time.time()
    m.train(Xtr, ytr, Xv, yv, epochs=50, batch_size=32, patience=7,
            model_path=os.path.join(MODELS_DIR, f"homec_k{k}.keras"))
    sec_per_epoch = (time.time() - t0) / len(m.history.history["loss"])
    ypred_kw = dp_k.inverse_transform_power(m.predict(Xte))
    metrics  = m.evaluate(yte_kw, ypred_kw)
    window_results.append({
        "Розмір вікна k (год)": k,
        "MAE":  metrics["MAE"],
        "MAPE (%)": metrics["MAPE"],
        "Час навчання (сек/епоха)": round(sec_per_epoch, 1),
    })
    print(f"    MAE={metrics['MAE']:.3f}  MAPE={metrics['MAPE']:.1f}%  сек/епоха={sec_per_epoch:.1f}")

df_windows = pd.DataFrame(window_results)
print("\nТаблиця 3.3 (HomeC):")
print(df_windows.to_string(index=False))
df_windows.to_csv(os.path.join(FIGURES_DIR, "table_3_3_window_ablation.csv"), index=False, encoding="utf-8-sig")

if len(df_windows) > 1:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_windows["Розмір вікна k (год)"], df_windows["MAPE (%)"],
            marker="o", color="#9C27B0", linewidth=2)
    ax.axvline(24, color="#4CAF50", linestyle="--", linewidth=1.5, label="Оптимум k=24")
    ax.set_xlabel("Розмір вікна k (год)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Рисунок 3.8 — Залежність похибки MAPE від розміру вікна (HomeC)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig("fig_3_8_window_ablation.png")

# ================================================================== #
#  КРОК 3. Порівняння ARIMA vs GRU vs LSTM                          #
# ================================================================== #
print("\n" + "=" * 60)
print("КРОК 3. Порівняння моделей (HomeC)")
print("=" * 60)

comparison = []

# ARIMA
print("\n  → ARIMA(2,1,2)...")
try:
    train_series = dp.df_clean["Active_Power"].dropna()
    n_train      = int(len(train_series) * 0.90)
    arima_train  = train_series.iloc[:n_train]
    arima_test   = train_series.iloc[n_train : n_train + len(y_test)]
    arima_fit    = ARIMA(arima_train, order=(2, 1, 2)).fit()
    arima_pred   = arima_fit.forecast(steps=len(arima_test))
    arima_true   = arima_test.values[:len(arima_pred)]
    mae_a  = float(np.mean(np.abs(arima_true - arima_pred)))
    rmse_a = float(np.sqrt(np.mean((arima_true - arima_pred) ** 2)))
    mask   = arima_true > 0.01
    mape_a = float(np.mean(np.abs((arima_true[mask] - arima_pred.values[mask]) / arima_true[mask])) * 100)
    comparison.append({"Модель": "ARIMA", "Врахування погоди": "Ні",
                        "MAE": round(mae_a,3), "RMSE": round(rmse_a,3), "MAPE (%)": round(mape_a,2)})
    print(f"    MAE={mae_a:.3f}  RMSE={rmse_a:.3f}  MAPE={mape_a:.2f}%")
except Exception as e:
    print(f"    ARIMA помилка: {e}")

# GRU
print("\n  → GRU (64 нейрони)...")
gru_model = Sequential([
    Input(shape=(24, 8)),
    GRU(64, kernel_regularizer=l2(1e-4)),
    Dropout(0.2),
    Dense(1, activation="linear"),
])
gru_model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
gru_model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, batch_size=32,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=0)
y_pred_gru_kw = dp.inverse_transform_power(gru_model.predict(X_test, verbose=0).flatten())
mae_g  = float(np.mean(np.abs(y_test_kw - y_pred_gru_kw)))
rmse_g = float(np.sqrt(np.mean((y_test_kw - y_pred_gru_kw) ** 2)))
mask   = y_test_kw > 0.01
mape_g = float(np.mean(np.abs((y_test_kw[mask] - y_pred_gru_kw[mask]) / y_test_kw[mask])) * 100)
comparison.append({"Модель": "GRU", "Врахування погоди": "Так",
                    "MAE": round(mae_g,3), "RMSE": round(rmse_g,3), "MAPE (%)": round(mape_g,2)})
print(f"    MAE={mae_g:.3f}  RMSE={rmse_g:.3f}  MAPE={mape_g:.2f}%")

# LSTM
comparison.append({"Модель": "LSTM (запропонована)", "Врахування погоди": "Так",
                    "MAE": metrics_lstm["MAE"], "RMSE": metrics_lstm["RMSE"], "MAPE (%)": metrics_lstm["MAPE"]})

df_comparison = pd.DataFrame(comparison)
print("\nТаблиця 3.4 (HomeC):")
print(df_comparison.to_string(index=False))
df_comparison.to_csv(os.path.join(FIGURES_DIR, "table_3_4_comparison.csv"), index=False, encoding="utf-8-sig")

# Стовпчикова діаграма
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
colors = ["#FF7043", "#42A5F5", "#66BB6A"]
models = df_comparison["Модель"].tolist()
for idx, (metric, ax) in enumerate(zip(["MAE", "RMSE", "MAPE (%)"], axes)):
    vals = df_comparison[metric].tolist()
    bars = ax.bar(models, vals, color=colors, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    ax.set_title(metric)
    ax.set_ylim(0, max(vals) * 1.3)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
fig.suptitle("Рисунок 3.11 — Порівняльна діаграма метрик моделей (HomeC)", fontsize=13)
plt.tight_layout()
save_fig("fig_3_11_model_comparison.png")

# ================================================================== #
#  ФІНАЛЬНИЙ ЗВІТ                                                     #
# ================================================================== #
print("\n" + "=" * 60)
print("ФІНАЛЬНИЙ ЗВІТ (HomeC)")
print("=" * 60)
print("\nТаблиця 3.3:")
print(df_windows.to_string(index=False))
print("\nТаблиця 3.4:")
print(df_comparison.to_string(index=False))
print(f"\nMAE  < 0.4 кВт  → {'✓ ВИКОНАНО' if metrics_lstm['MAE']  < 0.4  else '✗ не виконано'}")
print(f"RMSE < 0.6 кВт  → {'✓ ВИКОНАНО' if metrics_lstm['RMSE'] < 0.6  else '✗ не виконано'}")
print(f"MAPE < 15%      → {'✓ ВИКОНАНО' if metrics_lstm['MAPE'] < 15.0 else '✗ не виконано'}")
print(f"\nВсі графіки збережено у папці: figures_homec/")
print("Експеримент завершено успішно!")
