"""
DataProcessorHomeC — клас попереднього оброблення даних.
Джерело: Smart Home Dataset with Weather Information (HomeC.csv)
Відрізняється від DataProcessor наявністю реальних погодних даних
(температура, вологість) та виправленим індексом часу.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataProcessorHomeC:
    """
    Повний цикл підготовки даних HomeC.csv для LSTM-моделі.
    Виклик: load() → resample() → clean() → normalize() → make_windows()
    """

    def __init__(self, filepath: str, window_size: int = 24, overlap_ratio: float = 0.0):
        self.filepath      = filepath
        self.window_size   = window_size
        self.overlap_ratio = overlap_ratio

        self.df_raw        = None
        self.df_clean      = None
        self.df_normalized = None
        self.scaler        = MinMaxScaler(feature_range=(0, 1))

        # Реальні погодні ознаки замість синтетичних
        self.feature_columns = [
            "Active_Power", "Temperature", "Humidity",
            "Is_Weekend", "Hour_sin", "Hour_cos", "Month_sin", "Month_cos"
        ]

    # ------------------------------------------------------------------ #
    #  Крок 1. Завантаження                                               #
    # ------------------------------------------------------------------ #
    def load(self) -> "DataProcessorHomeC":
        print("[1/5] Завантаження HomeC.csv...")

        df = pd.read_csv(self.filepath, low_memory=False)

        # Колонка time у датасеті має баг (крок=1 сек замість 60 сек),
        # тому будуємо правильний datetime вручну: 1 хвилина на рядок
        df["Datetime"] = pd.date_range(
            start="2016-01-01 05:00:00",
            periods=len(df),
            freq="1min"
        )
        df.set_index("Datetime", inplace=True)

        # Вибираємо потрібні стовпці
        df = df[["use [kW]", "temperature", "humidity"]].copy()
        df.columns = ["Active_Power", "Temperature", "Humidity"]

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Температура у файлі — Фаренгейти → Цельсії
        df["Temperature"] = (df["Temperature"] - 32) * 5 / 9

        self.df_raw = df
        print(f"    Рядків завантажено : {len(df):,}")
        print(f"    Діапазон дат       : {df.index.min()} → {df.index.max()}")
        return self

    # ------------------------------------------------------------------ #
    #  Крок 2. Ресемплінг до 1 год + ознаки                              #
    # ------------------------------------------------------------------ #
    def resample(self) -> "DataProcessorHomeC":
        print("[2/5] Ресемплінг до 1 год...")

        df = self.df_raw.resample("1h").mean(numeric_only=True)

        df["Is_Weekend"] = (df.index.dayofweek >= 5).astype(float)
        df["Hour_sin"]   = np.sin(2 * np.pi * df.index.hour / 24)
        df["Hour_cos"]   = np.cos(2 * np.pi * df.index.hour / 24)
        df["Month_sin"]  = np.sin(2 * np.pi * (df.index.month - 1) / 12)
        df["Month_cos"]  = np.cos(2 * np.pi * (df.index.month - 1) / 12)

        self.df_clean = df
        print(f"    Рядків після ресемплінгу: {len(df):,}")
        return self

    # ------------------------------------------------------------------ #
    #  Крок 3. Очищення пропусків                                         #
    # ------------------------------------------------------------------ #
    def clean(self) -> "DataProcessorHomeC":
        print("[3/5] Очищення пропусків...")

        df = self.df_clean.copy()
        missing_before = df["Active_Power"].isna().sum()

        df["Active_Power"] = df["Active_Power"].interpolate(method="linear", limit=3)
        df["Active_Power"] = df["Active_Power"].ffill().bfill()

        for col in ["Temperature", "Humidity"]:
            df[col] = df[col].interpolate(method="linear", limit=6)
            df[col] = df[col].ffill().bfill()

        print(f"    Пропусків Active_Power: до={missing_before}, після={df['Active_Power'].isna().sum()}")
        self.df_clean = df
        return self

    # ------------------------------------------------------------------ #
    #  Крок 4. Нормалізація Min-Max                                       #
    # ------------------------------------------------------------------ #
    def normalize(self) -> "DataProcessorHomeC":
        print("[4/5] Нормалізація Min-Max...")
        df = self.df_clean[self.feature_columns].copy()
        normalized = self.scaler.fit_transform(df)
        self.df_normalized = pd.DataFrame(
            normalized, index=df.index, columns=self.feature_columns
        )
        print(f"    Ознаки: {self.feature_columns}")
        return self

    # ------------------------------------------------------------------ #
    #  Крок 5. Sliding Window → 3D тензор                                #
    # ------------------------------------------------------------------ #
    def make_windows(self):
        print("[5/5] Формування sliding window тензорів...")

        data       = self.df_normalized.values
        target_idx = self.feature_columns.index("Active_Power")
        step       = max(1, int(np.ceil(self.window_size * (1 - self.overlap_ratio))))

        X, y, timestamps = [], [], []
        i = 0
        while i + self.window_size < len(data):
            X.append(data[i : i + self.window_size])
            y.append(data[i + self.window_size, target_idx])
            timestamps.append(self.df_normalized.index[i + self.window_size])
            i += step

        X          = np.array(X, dtype=np.float32)
        y          = np.array(y, dtype=np.float32)
        timestamps = np.array(timestamps)

        print(f"    Тензор X : {X.shape}  (samples × timesteps × features)")
        print(f"    Вектор y : {y.shape}")

        n         = len(X)
        train_end = int(n * 0.80)   # 80% навчання як домовились
        val_end   = int(n * 0.90)   # 10% валідація, 10% тест

        X_train, y_train = X[:train_end],         y[:train_end]
        X_val,   y_val   = X[train_end:val_end],   y[train_end:val_end]
        X_test,  y_test  = X[val_end:],            y[val_end:]
        ts_test          = timestamps[val_end:]

        print(f"    Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
        return X_train, X_val, X_test, y_train, y_val, y_test, ts_test

    # ------------------------------------------------------------------ #
    #  Зворотна нормалізація                                              #
    # ------------------------------------------------------------------ #
    def inverse_transform_power(self, y_scaled: np.ndarray) -> np.ndarray:
        dummy = np.zeros((len(y_scaled), len(self.feature_columns)))
        dummy[:, 0] = y_scaled.flatten()
        return self.scaler.inverse_transform(dummy)[:, 0]


# ------------------------------------------------------------------ #
#  Перевірка                                                          #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "HomeC.csv")

    dp = DataProcessorHomeC(filepath=path, window_size=24, overlap_ratio=0.96)
    dp.load().resample().clean().normalize()
    X_train, X_val, X_test, y_train, y_val, y_test, ts_test = dp.make_windows()

    print("\n=== Перевірка ===")
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"Перший timestamp тесту: {ts_test[0]}")
    print("\nDataProcessorHomeC працює коректно!")