"""
experiments.py — головний скрипт навчання та експериментів.
Відповідає Розділу 3 методички:
  - п. 3.2: розбиття даних, пайплайн тестування
  - п. 3.3: абляція розміру вікна (Таблиця 3.3)
  - п. 3.4: порівняння ARIMA vs GRU vs LSTM (Таблиця 3.4)
  - п. 3.5: розрахунок економічної ефективності (Таблиця 3.5)

Запускати: відкрити у VS Code та натиснути Run (▶)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # прибираємо зайві логи TensorFlow

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # збереження графіків без відкриття вікна
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from ml.data_processor import DataProcessor
from ml.lstm_model import LSTMModel

# ------------------------------------------------------------------ #
#  Налаштування шляхів                                                #
# ------------------------------------------------------------------ #
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "household_power_consumption.txt")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
#  Допоміжна функція збереження графіків                              #
# ------------------------------------------------------------------ #
def save_fig(name: str) -> None:
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Збережено: figures/{name}")


# ================================================================== #
#  КРОК 0. Завантаження та підготовка даних                          #
# ================================================================== #
print("=" * 60)
print("КРОК 0. Підготовка даних")
print("=" * 60)

dp = DataProcessor(filepath=DATA_PATH, window_size=24, overlap_ratio=0.96)
dp.load().resample().clean().normalize()
X_train, X_val, X_test, y_train, y_val, y_test, ts_test = dp.make_windows()

# Зберігаємо ненормалізовані значення для метрик у кВт
y_test_kw = dp.inverse_transform_power(y_test)


# ================================================================== #
#  КРОК 1. Навчання основної LSTM-моделі (вікно 24 год)              #
# ================================================================== #
print("\n" + "=" * 60)
print("КРОК 1. Навчання LSTM-моделі (window=24, оптимум)")
print("=" * 60)

lstm = LSTMModel(window_size=24, n_features=8, lstm_units=64, dropout_rate=0.2)
lstm.build()
lstm.train(
    X_train, y_train,
    X_val,   y_val,
    epochs=100,
    batch_size=32,
    patience=10,
    model_path=os.path.join(MODELS_DIR, "lstm_best.keras"),
)

# Прогноз на тестовій вибірці
y_pred_lstm_norm = lstm.predict(X_test)
y_pred_lstm_kw   = dp.inverse_transform_power(y_pred_lstm_norm)
metrics_lstm     = lstm.evaluate(y_test_kw, y_pred_lstm_kw)
print(f"\nLSTM метрики на тесті: {metrics_lstm}")

# -- Графік 1: криві навчання (Рисунок 3.9) --
print("\nГенерація графіків навчання...")
hist = lstm.history.history
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hist["loss"],     label="Train Loss", color="#2196F3")
ax.plot(hist["val_loss"], label="Val Loss",   color="#F44336", linestyle="--")
ax.set_xlabel("Епоха")
ax.set_ylabel("MSE Loss")
ax.set_title("Рисунок 3.9 — Динаміка функції втрат під час навчання")
ax.legend()
ax.grid(True, alpha=0.3)
save_fig("fig_3_9_learning_curves.png")

# -- Графік 2: прогноз vs реальність (Рисунок 3.12) --
print("Генерація графіку прогнозу...")
n_show = min(168, len(y_test_kw))   # показуємо перші 168 год (тиждень)
dates  = pd.to_datetime(ts_test[:n_show])

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates, y_test_kw[:n_show],    label="Реальне споживання", color="#2196F3",  linewidth=1.5)
ax.plot(dates, y_pred_lstm_kw[:n_show], label="Прогноз LSTM",    color="#F44336",  linewidth=1.5, linestyle="--")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.xticks(rotation=45)
ax.set_xlabel("Дата")
ax.set_ylabel("Споживання (кВт)")
ax.set_title("Рисунок 3.12 — Фактичне споживання vs Прогноз LSTM")
ax.legend()
ax.grid(True, alpha=0.3)
save_fig("fig_3_12_forecast_vs_real.png")

# -- Графік 3: гістограма залишків (Рисунок 3.13) --
residuals = y_test_kw - y_pred_lstm_kw
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(residuals, bins=40, color="#4CAF50", edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Нуль")
ax.set_xlabel("Залишкова похибка (кВт)")
ax.set_ylabel("Частота")
ax.set_title("Рисунок 3.13 — Гістограма залишкових похибок")
ax.legend()
ax.grid(True, alpha=0.3)
save_fig("fig_3_13_residuals.png")


# ================================================================== #
#  КРОК 2. Абляція розміру вікна (Таблиця 3.3, Рисунок 3.8)         #
# ================================================================== #
print("\n" + "=" * 60)
print("КРОК 2. Абляція розміру вікна (12 / 24 / 48 / 168)")
print("=" * 60)

window_results = []

for k in [12, 24, 48, 168]:
    print(f"\n  → Вікно k={k} год...")
    dp_k = DataProcessor(filepath=DATA_PATH, window_size=k)
    dp_k.load().resample().clean().normalize()
    Xtr, Xv, Xte, ytr, yv, yte, _ = dp_k.make_windows()

    if len(Xtr) < 10:
        print(f"    Недостатньо даних для k={k}, пропускаємо.")
        continue

    yte_kw = dp_k.inverse_transform_power(yte)

    m = LSTMModel(window_size=k, n_features=8, lstm_units=64)
    m.build()

    import time
    t0 = time.time()
    m.train(Xtr, ytr, Xv, yv, epochs=50, batch_size=32, patience=7,
            model_path=os.path.join(MODELS_DIR, f"lstm_k{k}.keras"))
    sec_per_epoch = (time.time() - t0) / len(m.history.history["loss"])

    ypred_kw = dp_k.inverse_transform_power(m.predict(Xte))
    metrics  = m.evaluate(yte_kw, ypred_kw)

    window_results.append({
        "Розмір вікна k (год)": k,
        "MAE":  metrics["MAE"],
        "MAPE (%)": metrics["MAPE"],
        "Час навчання (сек/епоха)": round(sec_per_epoch, 1),
    })
    print(f"    MAE={metrics['MAE']:.3f}  MAPE={metrics['MAPE']:.1f}%  "
          f"сек/епоха={sec_per_epoch:.1f}")

df_windows = pd.DataFrame(window_results)
print("\nТаблиця 3.3 — Результати абляції вікна:")
print(df_windows.to_string(index=False))
df_windows.to_csv(os.path.join(FIGURES_DIR, "table_3_3_window_ablation.csv"), index=False, encoding="utf-8-sig")

# -- Графік: залежність MAPE від розміру вікна (Рисунок 3.8) --
if len(df_windows) > 1:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_windows["Розмір вікна k (год)"], df_windows["MAPE (%)"],
            marker="o", color="#9C27B0", linewidth=2)
    ax.axvline(24, color="#4CAF50", linestyle="--", linewidth=1.5, label="Оптимум k=24")
    ax.set_xlabel("Розмір вікна k (год)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Рисунок 3.8 — Залежність похибки MAPE від розміру вікна")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig("fig_3_8_window_ablation.png")

# ================================================================== #
#  КРОК 2.5. Абляція кількості нейронів (Рисунок 3.10)               #
# ================================================================== #
print("\n" + "=" * 60)
print("КРОК 2.5. Тестування кількості нейронів LSTM-шару")
print("=" * 60)

neuron_results = []

for units in [16, 32, 64, 128, 256]:
    print(f"\n  → {units} нейронів...")
    m = LSTMModel(window_size=24, n_features=8, lstm_units=units)
    m.build()
    m.train(X_train, y_train, X_val, y_val,
            epochs=50, batch_size=32, patience=7,
            model_path=os.path.join(MODELS_DIR, f"lstm_u{units}.keras"))
    ypred = dp.inverse_transform_power(m.predict(X_test))
    met   = m.evaluate(y_test_kw, ypred)
    neuron_results.append({"Нейронів": units, "MAE": met["MAE"], "MAPE (%)": met["MAPE"]})
    print(f"    MAE={met['MAE']:.3f}  MAPE={met['MAPE']:.1f}%")

df_neurons = pd.DataFrame(neuron_results)
print("\nТаблиця — Абляція кількості нейронів:")
print(df_neurons.to_string(index=False))
df_neurons.to_csv(os.path.join(FIGURES_DIR, "table_neurons_ablation.csv"),
                  index=False, encoding="utf-8-sig")

# -- Графік: діаграма розсіювання (Рисунок 3.10) --
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(df_neurons["Нейронів"], df_neurons["MAPE (%)"],
           color="#9C27B0", s=120, zorder=5)
ax.plot(df_neurons["Нейронів"], df_neurons["MAPE (%)"],
        color="#9C27B0", linewidth=1.5, linestyle="--", alpha=0.6)
best_row = df_neurons.loc[df_neurons["MAPE (%)"].idxmin()]
ax.axvline(best_row["Нейронів"], color="#4CAF50", linestyle="--",
           linewidth=1.5, label=f'Оптимум: {int(best_row["Нейронів"])} нейронів')
ax.set_xlabel("Кількість нейронів LSTM-шару")
ax.set_ylabel("MAPE (%)")
ax.set_title("Рисунок 3.10 — Діаграма розсіювання точності відносно кількості нейронів")
ax.legend()
ax.grid(True, alpha=0.3)
save_fig("fig_3_10_neurons_ablation.png")

# ================================================================== #
#  КРОК 3. Порівняння ARIMA vs GRU vs LSTM (Таблиця 3.4)            #
# ================================================================== #
print("\n" + "=" * 60)
print("КРОК 3. Порівняння моделей: ARIMA vs GRU vs LSTM")
print("=" * 60)

comparison = []

# --- ARIMA ---
print("\n  → ARIMA(2,1,2)...")
try:
    # Для ARIMA використовуємо лише одновимірний ряд (без погоди)
    train_series = dp.df_clean["Active_Power"].dropna()
    n_train = int(len(train_series) * 0.85)
    arima_train = train_series.iloc[:n_train]
    arima_test  = train_series.iloc[n_train : n_train + len(y_test)]

    arima_model = ARIMA(arima_train, order=(2, 1, 2))
    arima_fit   = arima_model.fit()
    arima_pred  = arima_fit.forecast(steps=len(arima_test))

    arima_true = arima_test.values[:len(arima_pred)]
    mae_a  = float(np.mean(np.abs(arima_true - arima_pred)))
    rmse_a = float(np.sqrt(np.mean((arima_true - arima_pred) ** 2)))
    mask   = arima_true > 0.01
    mape_a = float(np.mean(np.abs((arima_true[mask] - arima_pred.values[mask])
                                   / arima_true[mask])) * 100)

    comparison.append({
        "Модель": "ARIMA",
        "Врахування погоди": "Ні",
        "MAE":  round(mae_a, 3),
        "RMSE": round(rmse_a, 3),
        "MAPE (%)": round(mape_a, 2),
    })
    print(f"    MAE={mae_a:.3f}  RMSE={rmse_a:.3f}  MAPE={mape_a:.2f}%")
except Exception as e:
    print(f"    ARIMA помилка: {e}")

# --- GRU ---
print("\n  → GRU (64 нейрони)...")
gru_model = Sequential([
    Input(shape=(24, 8)),
    GRU(64, kernel_regularizer=l2(1e-4)),
    Dropout(0.2),
    Dense(1, activation="linear"),
], name="GRU_Baseline")
gru_model.compile(optimizer=Adam(0.001), loss="mse", metrics=["mae"])
gru_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=0,
)
y_pred_gru_kw = dp.inverse_transform_power(gru_model.predict(X_test, verbose=0).flatten())
mae_g  = float(np.mean(np.abs(y_test_kw - y_pred_gru_kw)))
rmse_g = float(np.sqrt(np.mean((y_test_kw - y_pred_gru_kw) ** 2)))
mask   = y_test_kw > 0.01
mape_g = float(np.mean(np.abs((y_test_kw[mask] - y_pred_gru_kw[mask])
                               / y_test_kw[mask])) * 100)
comparison.append({
    "Модель": "GRU",
    "Врахування погоди": "Так",
    "MAE":  round(mae_g, 3),
    "RMSE": round(rmse_g, 3),
    "MAPE (%)": round(mape_g, 2),
})
print(f"    MAE={mae_g:.3f}  RMSE={rmse_g:.3f}  MAPE={mape_g:.2f}%")

# --- LSTM (вже навчена вище) ---
comparison.append({
    "Модель": "LSTM (запропонована)",
    "Врахування погоди": "Так",
    "MAE":  metrics_lstm["MAE"],
    "RMSE": metrics_lstm["RMSE"],
    "MAPE (%)": metrics_lstm["MAPE"],
})

df_comparison = pd.DataFrame(comparison)
print("\nТаблиця 3.4 — Порівняльний аналіз моделей:")
print(df_comparison.to_string(index=False))
df_comparison.to_csv(os.path.join(FIGURES_DIR, "table_3_4_comparison.csv"), index=False, encoding="utf-8-sig")

# -- Графік: стовпчикова діаграма метрик (Рисунок 3.11) --
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
colors = ["#FF7043", "#42A5F5", "#66BB6A"]
models = df_comparison["Модель"].tolist()

for idx, (metric, ax) in enumerate(zip(["MAE", "RMSE", "MAPE (%)"], axes)):
    vals = df_comparison[metric].tolist()
    bars = ax.bar(models, vals, color=colors, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(vals) * 1.3)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)

fig.suptitle("Рисунок 3.11 — Порівняльна діаграма метрик моделей", fontsize=13)
plt.tight_layout()
save_fig("fig_3_11_model_comparison.png")


# ================================================================== #
#  КРОК 4. Економічна ефективність (Таблиця 3.5, Рисунок 3.15)      #
# ================================================================== #
print("\n" + "=" * 60)
print("КРОК 4. Аналіз економічної ефективності")
print("=" * 60)

# Двозонний тариф (грн/кВт·год): пік 07:00–23:00, ніч 23:00–07:00
PEAK_TARIFF  = 4.32   # грн/кВт·год (денна зона)
NIGHT_TARIFF = 2.16   # грн/кВт·год (нічна зона, -50%)
HOURS_IN_DAY = 24

hours = np.arange(HOURS_IN_DAY)
tariffs = np.where((hours >= 7) & (hours < 23), PEAK_TARIFF, NIGHT_TARIFF)

# Типовий добовий профіль (на основі тестових даних)
daily_profile = np.zeros(HOURS_IN_DAY)
count_hours   = np.zeros(HOURS_IN_DAY)
for i, ts in enumerate(ts_test):
    h = pd.Timestamp(ts).hour
    daily_profile[h] += y_test_kw[i]
    count_hours[h]   += 1
count_hours[count_hours == 0] = 1
daily_profile /= count_hours   # середнє споживання по годинах

# Без оптимізації
cost_before = float(np.sum(daily_profile * tariffs))

# З оптимізацією: переносимо 25% пікового споживання на ніч
optimized = daily_profile.copy()
peak_mask   = (hours >= 18) & (hours < 22)   # вечірній пік
night_mask  = (hours >= 23) | (hours < 6)    # нічна зона

shift_amount = optimized[peak_mask] * 0.25
optimized[peak_mask]  -= shift_amount
optimized[night_mask] += shift_amount.sum() / night_mask.sum()

cost_after  = float(np.sum(optimized * tariffs))
saving_pct  = (cost_before - cost_after) / cost_before * 100

print(f"  Вартість до оптимізації (доба): {cost_before:.2f} грн")
print(f"  Вартість після оптимізації     : {cost_after:.2f} грн")
print(f"  Економія                        : {saving_pct:.1f}%")
print(f"  Орієнтовна економія за місяць  : {(cost_before - cost_after) * 30:.0f} грн")

# -- Графік: профіль навантаження до/після (Рисунок 3.15) --
fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(hours, daily_profile, alpha=0.25, color="#F44336")
ax.fill_between(hours, optimized,     alpha=0.25, color="#4CAF50")
ax.plot(hours, daily_profile, label="До оптимізації",    color="#F44336", linewidth=2, marker="o", markersize=4)
ax.plot(hours, optimized,     label="Після оптимізації", color="#4CAF50", linewidth=2, marker="s", markersize=4)
ax.axvspan(7, 23, alpha=0.05, color="orange", label="Пікова зона (07–23)")
ax.set_xlabel("Година доби")
ax.set_ylabel("Середнє споживання (кВт)")
ax.set_title("Рисунок 3.15 — Профіль навантаження до та після оптимізації")
ax.set_xticks(hours)
ax.legend()
ax.grid(True, alpha=0.3)
save_fig("fig_3_15_load_profile.png")


# ================================================================== #
#  ФІНАЛЬНИЙ ЗВІТ                                                     #
# ================================================================== #
print("\n" + "=" * 60)
print("ФІНАЛЬНИЙ ЗВІТ ЕКСПЕРИМЕНТУ")
print("=" * 60)
print("\nТаблиця 3.3 — Абляція розміру вікна:")
print(df_windows.to_string(index=False))
print("\nТаблиця 3.4 — Порівняння моделей:")
print(df_comparison.to_string(index=False))
print("\nЦільові бенчмарки (Таблиця 2.10):")
print(f"  MAE  < 0.4 кВт  → {'✓ ВИКОНАНО' if metrics_lstm['MAE']  < 0.4  else '✗ не виконано'}")
print(f"  RMSE < 0.6 кВт  → {'✓ ВИКОНАНО' if metrics_lstm['RMSE'] < 0.6  else '✗ не виконано'}")
print(f"  MAPE < 15%      → {'✓ ВИКОНАНО' if metrics_lstm['MAPE'] < 15.0 else '✗ не виконано'}")
print(f"\nВсі графіки збережено у папці: figures/")
print(f"Модель збережено у папці      : models/")
print("\nЕксперимент завершено успішно!")