"""
generate_ch2_plots.py — генерація рисунків 2.9, 2.10, 2.11 для розділу 2.
Запускати: py generate_ch2_plots.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "household_power_consumption.txt")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def save_fig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Збережено: figures/{name}")

# ------------------------------------------------------------------ #
#  Завантаження та підготовка даних                                   #
# ------------------------------------------------------------------ #
print("Завантаження даних...")
df = pd.read_csv(DATA_PATH, sep=";", na_values=["?"], low_memory=False)
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
df.set_index("Datetime", inplace=True)
df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

# Ресемплінг до 1 год
df_hourly = df[["Global_active_power"]].resample("1h").mean()
df_hourly.columns = ["Active_Power"]

# Синтетична температура (як у data_processor.py)
hours  = df_hourly.index.hour
months = df_hourly.index.month
seasonal = 8  * np.sin(2 * np.pi * (months - 3) / 12)
diurnal  = 4  * np.sin(2 * np.pi * (hours  - 6) / 24)
df_hourly["Temperature"] = np.round(seasonal + diurnal + 11, 1)
df_hourly["Humidity"]    = np.round(
    0.70 - 0.10 * np.sin(2 * np.pi * (months - 1) / 12)
         - 0.05 * np.sin(2 * np.pi * (hours  - 6) / 24), 2
)
df_hourly = df_hourly.dropna()
print(f"Годинних записів: {len(df_hourly):,}")

# ================================================================== #
#  Рисунок 2.9 — Візуалізація неочищеного часового ряду              #
# ================================================================== #
print("\nГенерація рисунку 2.9...")

# Беремо перший місяць для наочності
df_raw_month = df_hourly["Active_Power"].iloc[:720]  # ~30 днів

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df_raw_month.index, df_raw_month.values,
        color="#2196F3", linewidth=0.8, alpha=0.85)

# Позначаємо кілька аномальних піків
peaks_idx = df_raw_month.nlargest(3).index
ax.scatter(peaks_idx, df_raw_month[peaks_idx],
           color="#F44336", zorder=5, s=40, label="Аномальні піки")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=30)
ax.set_xlabel("Дата")
ax.set_ylabel("Активна потужність (кВт)")
ax.set_title("Рисунок 2.9 — Візуалізація неочищеного часового ряду споживання електроенергії")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
save_fig("fig_2_9_raw_timeseries.png")

# ================================================================== #
#  Рисунок 2.10 — Гістограма розподілу значень активної потужності   #
# ================================================================== #
print("Генерація рисунку 2.10...")

fig, ax = plt.subplots(figsize=(9, 4))
n, bins, patches = ax.hist(
    df_hourly["Active_Power"].dropna(),
    bins=60, color="#4CAF50", edgecolor="white",
    linewidth=0.5, alpha=0.88
)

# Позначаємо медіану та середнє
mean_val   = df_hourly["Active_Power"].mean()
median_val = df_hourly["Active_Power"].median()
ax.axvline(mean_val,   color="#F44336", linestyle="--", linewidth=1.5,
           label=f"Середнє: {mean_val:.2f} кВт")
ax.axvline(median_val, color="#FF9800", linestyle=":",  linewidth=1.5,
           label=f"Медіана: {median_val:.2f} кВт")

ax.set_xlabel("Активна потужність (кВт)")
ax.set_ylabel("Частота (кількість годин)")
ax.set_title("Рисунок 2.10 — Гістограма розподілу значень активної потужності")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
save_fig("fig_2_10_histogram.png")

# ================================================================== #
#  Рисунок 2.11 — Матриця кореляції Пірсона                         #
# ================================================================== #
print("Генерація рисунку 2.11...")

# Додаємо часові ознаки для матриці кореляції
df_corr = df_hourly[["Active_Power", "Temperature", "Humidity"]].copy()
df_corr["Година доби"]   = df_hourly.index.hour
df_corr["День тижня"]    = df_hourly.index.dayofweek
df_corr["Місяць"]        = df_hourly.index.month
df_corr = df_corr.dropna()

# Перейменовуємо для кращого відображення
df_corr.columns = [
    "Активна\nпотужність",
    "Температура",
    "Вологість",
    "Година\nдоби",
    "День\nтижня",
    "Місяць",
]

corr_matrix = df_corr.corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    vmin=-1, vmax=1,
    ax=ax,
    linewidths=0.5,
    linecolor="white",
    annot_kws={"size": 10},
    square=True,
)
ax.set_title("Рисунок 2.11 — Матриця кореляції Пірсона для вхідних ознак", pad=14)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
save_fig("fig_2_11_correlation_matrix.png")

print("\nУсі рисунки згенеровано успішно!")
print("Файли знаходяться у папці figures/:")
print("  fig_2_9_raw_timeseries.png")
print("  fig_2_10_histogram.png")
print("  fig_2_11_correlation_matrix.png")
