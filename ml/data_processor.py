"""
DataProcessor — клас попереднього оброблення даних.
Джерело: Individual Household Electric Power Consumption (household_power_consumption.txt)
Відповідає вимогам методички: Таблиця 2.1, формули 2.4–2.7, 2.14, Таблиця 2.7.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    """
    Повний цикл підготовки даних для LSTM-моделі.
    Методи викликаються послідовно:
        load() → resample() → clean() → normalize() → make_windows()
    """

    def __init__(self, filepath: str, window_size: int = 24, overlap_ratio: float = 0.0):
        """
        filepath      — шлях до датасету household_power_consumption.txt
        window_size   — розмір ковзного вікна k (за замовчуванням 24 год = доба)
        overlap_ratio — частка перекриття між сусідніми вікнами (0.0 = без перекриття)
        """
        self.filepath = filepath
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio

        self.df_raw        = None   #сирі дані після завантаження
        self.df_clean      = None   #після очищення та ресемплінгу
        self.df_normalized = None   #нормалізовані дані

        #Scaler зберігаємо, щоб потім повернути прогноз у кВт
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        #ознаки, що подаються на вхід мережі (Таблиця 2.1)
        self.feature_columns = ["Active_Power", "Temperature", "Humidity", "Is_Weekend", "Hour_sin", "Hour_cos", "Month_sin", "Month_cos"]

    #крок 1 - завантаження

    def load(self) -> "DataProcessor":
        """
        Читає household_power_consumption.txt, об'єднує стовпці
        Date і Time в єдиний DatetimeIndex, залишає Active_Power.
        """
        print("[1/5] Завантаження даних...")

        df = pd.read_csv(
            self.filepath,
            sep=";",
            na_values=["?"],
            low_memory=False,
        )

        #об'єднуємо Date + Time => єдина колонка datetime
        df["Datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"], dayfirst=True
        )
        df.set_index("Datetime", inplace=True)
        df.drop(columns=["Date", "Time"], inplace=True)

        #залишаємо лише активну потужність (кВт)
        df = df[["Global_active_power"]].rename(
            columns={"Global_active_power": "Active_Power"}
        )
        df["Active_Power"] = pd.to_numeric(df["Active_Power"], errors="coerce")

        self.df_raw = df
        print(f"    Рядків завантажено : {len(df):,}")
        print(f"    Діапазон дат       : {df.index.min()} → {df.index.max()}")
        return self

    #крок 2 - ресемплінг до 1 год + синтез погодних ознак
    
    def resample(self) -> "DataProcessor":
        """
        Усереднює хвилинні записи до 1-годинних інтервалів.
        Додає синтетичну температуру (сезонна + добова компонента),
        синтетичну вологість та бінарну ознаку Is_Weekend.

        Примітка: для підвищення точності можна замінити синтетичні
        погодні дані на реальні з відповідного регіону та часового
        діапазону датасету (Франція, 2006–2010).
        """
        print("[2/5] Ресемплінг до 1 год та формування ознак...")

        df = self.df_raw.resample("1h").mean(numeric_only=True)

        hours  = df.index.hour
        months = df.index.month

        #синтетична температура (*C): сезонна + добова компонента
        #базована на кліматі Парижа (де розташований датасет UCI)
        seasonal = 8  * np.sin(2 * np.pi * (months - 3) / 12)   # -8..+8 *C сезонно
        diurnal  = 4  * np.sin(2 * np.pi * (hours  - 6) / 24)   # ±4 *C протягом доби
        df["Temperature"] = np.round(seasonal + diurnal + 11, 1) # середня ~11*C (Париж)

        #синтетична вологість (%): вища вночі та взимку
        df["Humidity"] = np.round(
            0.70 - 0.10 * np.sin(2 * np.pi * (months - 1) / 12)
                 - 0.05 * np.sin(2 * np.pi * (hours  - 6) / 24), 2
        )

        #бінарна ознака вихідного дня таблиця 2.1: D_t)
        df["Is_Weekend"] = (df.index.dayofweek >= 5).astype(float)
        #циклічні часові ознаки (sin/cos години та місяця)
        df["Hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["Hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        df["Month_sin"] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
        df["Month_cos"] = np.cos(2 * np.pi * (df.index.month - 1) / 12)

        self.df_clean = df
        print(f"    Рядків після ресемплінгу: {len(df):,}")
        return self

    #крок 3 - очищення пропущених значень таблиця 2.7)
    
    def clean(self) -> "DataProcessor":
        """
        Заповнює пропуски:
          - до 3 підряд → лінійна інтерполяція
          - більше 3    → forward fill (значення попередньої доби)
        """
        print("[3/5] Очищення пропусків...")

        df = self.df_clean.copy()
        missing_before = df["Active_Power"].isna().sum()

        df["Active_Power"] = df["Active_Power"].interpolate(method="linear", limit=3)
        df["Active_Power"] = df["Active_Power"].ffill()
        df["Active_Power"] = df["Active_Power"].bfill()

        missing_after = df["Active_Power"].isna().sum()
        print(f"    Пропусків Active_Power: до={missing_before}, після={missing_after}")

        self.df_clean = df
        return self

    #крок 4 - нормалізація Min-Max (формула 2.14)
    
    def normalize(self) -> "DataProcessor":
        """
        Приводить усі ознаки до діапазону [0; 1].
        Scaler зберігається для зворотного перетворення прогнозу.
        """
        print("[4/5] Нормалізація Min-Max...")

        df = self.df_clean[self.feature_columns].copy()
        normalized = self.scaler.fit_transform(df)
        self.df_normalized = pd.DataFrame(
            normalized, index=df.index, columns=self.feature_columns
        )
        print(f"    Ознаки: {self.feature_columns}")
        return self

    #крок 5 - Sliding Window => 3D тензор (формули 2.4–2.7)

    def make_windows(self):
        """
        Перетворює нормалізований часовий ряд у набір ковзних вікон.

        Повертає (кортеж із 7 елементів):
            X_train, X_val, X_test  — форма (samples, window_size, features)
            y_train, y_val, y_test  — форма (samples,)
            timestamps_test         — масив datetime для тестової вибірки
        """
        print("[5/5] Формування sliding window тензорів...")

        data       = self.df_normalized.values
        target_idx = self.feature_columns.index("Active_Power")

        #крок зсуву вікна (формула 2.6)
        step = max(1, int(np.ceil(self.window_size * (1 - self.overlap_ratio))))

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

        #розбиття 70 % / 15 % / 15 % (п. 3.2 методички)
        n         = len(X)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)

        X_train, y_train = X[:train_end],         y[:train_end]
        X_val,   y_val   = X[train_end:val_end],   y[train_end:val_end]
        X_test,  y_test  = X[val_end:],            y[val_end:]
        ts_test          = timestamps[val_end:]

        print(f"    Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
        return X_train, X_val, X_test, y_train, y_val, y_test, ts_test

    #допоміжний метод: зворотна нормалізація прогнозу
    
    def inverse_transform_power(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Перетворює нормалізований прогноз назад у кВт.
        Приймає масив форми (N,) або (N, 1).
        """
        dummy = np.zeros((len(y_scaled), len(self.feature_columns)))
        dummy[:, 0] = y_scaled.flatten()
        return self.scaler.inverse_transform(dummy)[:, 0]


#швидка перевірка (запускати напряму через Run у VS Code)

if __name__ == "__main__":
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "household_power_consumption.txt")

    dp = DataProcessor(filepath=path, window_size=24)
    dp.load().resample().clean().normalize()
    X_train, X_val, X_test, y_train, y_val, y_test, ts_test = dp.make_windows()

    print("\n=== Перевірка форм тензорів ===")
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"Перший timestamp тесту: {ts_test[0]}")
    print("\nDataProcessor працює коректно!")