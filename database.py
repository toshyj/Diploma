"""
database.py — модуль роботи з базою даних SQLite.
Таблиці відповідають Таблиці 2.9 та ER-діаграмі методички.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db")

def get_connection():
    """Повертає з'єднання з БД."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # результати як словники
    return conn


def init_db():
    """
    Створює таблиці якщо вони ще не існують.
    Викликати один раз при старті застосунку.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Таблиця вимірювань (Таблиця 2.9 методички)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Measurements (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    DATETIME NOT NULL,
            active_power REAL     NOT NULL,
            temperature  REAL,
            humidity     REAL,
            dataset      TEXT     NOT NULL DEFAULT 'uci'
        )
    """)

    # Таблиця прогнозів
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Predictions (
            id              INTEGER  PRIMARY KEY AUTOINCREMENT,
            created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            dataset         TEXT     NOT NULL,
            period_label    TEXT     NOT NULL,
            mae             REAL,
            rmse            REAL,
            mape            REAL
        )
    """)

    # Таблиця сесій користувача
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS UserSessions (
            id          INTEGER  PRIMARY KEY AUTOINCREMENT,
            created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            dataset     TEXT     NOT NULL,
            period_label TEXT    NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print(f"БД ініціалізована: {DB_PATH}")


def save_prediction_session(dataset, period_label, mae, rmse, mape, created_at=None):
    conn = get_connection()
    conn.execute(
        """INSERT INTO Predictions (dataset, period_label, mae, rmse, mape, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (dataset, period_label, mae, rmse, mape, created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

def save_user_session(dataset, period_label, created_at=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO UserSessions (dataset, period_label, created_at) VALUES (?, ?, ?)",
        (dataset, period_label, created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()


def get_recent_sessions(limit: int = 10) -> list:
    conn = get_connection()
    rows = conn.execute(
        """SELECT dataset, period_label, mae, rmse, mape, created_at
           FROM Predictions
           ORDER BY created_at DESC LIMIT ?""",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_prediction_stats() -> dict:
    """Повертає зведену статистику по збережених прогнозах."""
    conn = get_connection()
    row = conn.execute(
        """SELECT COUNT(*) as total,
                  AVG(mae)  as avg_mae,
                  AVG(rmse) as avg_rmse,
                  AVG(mape) as avg_mape
           FROM Predictions"""
    ).fetchone()
    conn.close()
    if row and row["total"]:
        return {
            "total":    row["total"],
            "avg_mae":  round(row["avg_mae"],  4),
            "avg_rmse": round(row["avg_rmse"], 4),
            "avg_mape": round(row["avg_mape"], 2),
        }
    return {"total": 0, "avg_mae": 0, "avg_rmse": 0, "avg_mape": 0}


if __name__ == "__main__":
    init_db()
    print("Таблиці створено успішно!")

    # Тестовий запис
    save_user_session("uci", "Зима 2010")
    save_prediction_session("uci", "Зима 2010", 0.32, 0.47, 32.1)

    stats = get_prediction_stats()
    print(f"Статистика: {stats}")

    sessions = get_recent_sessions()
    print(f"Останні сесії: {sessions}")
