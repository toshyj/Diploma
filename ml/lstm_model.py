"""
LSTMModel — клас побудови, навчання та оцінювання нейронної мережі.
Архітектура відповідає Таблиці 2.4 методички.
Метрики оцінювання — формули 2.15, 2.16, 2.17.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

class LSTMModel:
    """
    Інкапсулює логіку побудови, навчання та збереження LSTM-моделі.
    """

    def __init__(
        self,
        window_size: int = 24,
        n_features: int = 8,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        l2_lambda: float = 1e-4,
    ):
        """
        window_size   — кількість часових кроків у вхідному вікні (k)
        n_features    — кількість ознак (F): Active_Power, Temp, Humidity, Is_Weekend
        lstm_units    — кількість нейронів у LSTM-шарі (Таблиця 2.4: 64)
        dropout_rate  — частка вимкнених зв'язків Dropout (Таблиця 2.4: 0.2)
        learning_rate — крок навчання оптимізатора Adam
        l2_lambda     — коефіцієнт L2-регуляризації (формула 2.13)
        """
        self.window_size   = window_size
        self.n_features    = n_features
        self.lstm_units    = lstm_units
        self.dropout_rate  = dropout_rate
        self.learning_rate = learning_rate
        self.l2_lambda     = l2_lambda

        self.model   = None   #об'єкт Keras-моделі
        self.history = None   #історія навчання (для графіків розділу 3)

    #побудова архітектури (таблиця 2.4)

    def build(self) -> "LSTMModel":
        """
        Будує архітектуру:
          Input → LSTM(64, tanh/sigmoid) → Dropout(0.2) → Dense(1, linear)
        """
        self.model = Sequential([
            Input(shape=(self.window_size, self.n_features)),

            #LSTM-шар: виявляє приховані часові патерни (формули 2.9–2.11)
            LSTM(
                units=self.lstm_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                kernel_regularizer=l2(self.l2_lambda),
                return_sequences=False,
            ),

            #Dropout: регуляризація — вимикає 20 % зв'язків (таблиця 2.4)
            Dropout(self.dropout_rate),

            #вихідний шар регресії: одне значення — прогноз потужності
            Dense(1, activation="linear"),
        ], name="LSTM_Energy_Forecaster")

        #оптимізатор Adam (формула 2.12), функція втрат MSE (формула 2.13)
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        print("=== Архітектура моделі ===")
        self.model.summary()
        return self

    #начання

    def train(
        self,
        X_train, y_train,
        X_val,   y_val,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        model_path: str = "models/best_model.keras",
    ) -> "LSTMModel":
        """
        Навчає модель із механізмом Early Stopping (формула 3.2).

        epochs      — максимальна кількість епох (Таблиця 2.5: 100)
        batch_size  — розмір міні-батчу (Таблиця 2.5: 32)
        patience    — кількість епох очікування без покращення (P у формулі 3.2)
        model_path  — шлях для збереження найкращих ваг
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        callbacks = [
            #зупинка якщо val_loss не покращується P епох підряд (формула 3.2)
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            #зберігаємо найкращу модель на диск
            ModelCheckpoint(
                filepath=model_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ]

        print(f"\nНавчання моделі (max {epochs} епох, batch={batch_size})...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        best_epoch = np.argmin(self.history.history["val_loss"]) + 1
        best_val   = min(self.history.history["val_loss"])
        print(f"\nНавчання завершено. Найкраща епоха: {best_epoch}, val_loss: {best_val:.6f}")
        return self

    #прогнозування

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Повертає прогноз у нормалізованому масштабі (форма: (N,)).
        Для отримання значень у кВт використовуй DataProcessor.inverse_transform_power()
        """
        return self.model.predict(X, verbose=0).flatten()

    #метрики оцінювання (формули 2.15, 2.16, 2.17)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Обчислює MAE, RMSE, MAPE у реальних одиницях (кВт / %).
        y_true та y_pred мають бути вже в кВт (після inverse_transform).

        Повертає словник із результатами для Таблиці 3.4 методички.
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        #MAE — середня абсолютна похибка (формула 2.15)
        mae = float(np.mean(np.abs(y_true - y_pred)))

        #RMSE — середньоквадратична похибка (формула 2.16)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        #MAPE — середня абсолютна відсоткова похибка (формула 2.17)
        #sMAPE — симетрична MAPE, стійка до значень близьких до нуля
        mask = y_true > 0.5
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

        metrics = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2)}
        return metrics

    #збереження та завантаження

    def save(self, path: str = "models/best_model.keras") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Модель збережено: {path}")

    def load(self, path: str = "models/best_model.keras") -> "LSTMModel":
        self.model = load_model(path)
        print(f"Модель завантажено: {path}")
        return self


# ------------------------------------------------------------------ #
#  Швидка перевірка архітектури (запускати напряму у VS Code)        #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    model = LSTMModel(window_size=24, n_features=4)
    model.build()
    print("\nАрхітектура побудована успішно!")