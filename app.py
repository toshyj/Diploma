"""
app.py — Flask REST API з двома режимами прогнозу, вибором підвибірок та БД.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from ml.data_processor import DataProcessor
from ml.data_processor_homec import DataProcessorHomeC
from ml.lstm_model import LSTMModel
from database import init_db, save_user_session, save_prediction_session, get_recent_sessions, get_prediction_stats

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ #
#  Завантаження обох моделей при старті                               #
# ------------------------------------------------------------------ #
print("Ініціалізація бази даних...")
init_db()

print("\nЗавантаження UCI моделі...")
dp_uci = DataProcessor(
    filepath=os.path.join(BASE_DIR, "data", "household_power_consumption.txt"),
    window_size=24, overlap_ratio=0.96
)
dp_uci.load().resample().clean().normalize()
_, _, X_test_uci, _, _, y_test_uci, ts_test_uci = dp_uci.make_windows()
y_test_uci_kw = dp_uci.inverse_transform_power(y_test_uci)

lstm_uci = LSTMModel(window_size=24, n_features=8)
lstm_uci.load(os.path.join(BASE_DIR, "models", "lstm_best.keras"))
y_pred_uci_kw = dp_uci.inverse_transform_power(lstm_uci.predict(X_test_uci))
metrics_uci   = lstm_uci.evaluate(y_test_uci_kw, y_pred_uci_kw)
print(f"UCI готова. Метрики: {metrics_uci}")

print("\nЗавантаження HomeC моделі...")
dp_homec = DataProcessorHomeC(
    filepath=os.path.join(BASE_DIR, "data", "HomeC.csv"),
    window_size=24, overlap_ratio=0.96
)
dp_homec.load().resample().clean().normalize()
_, _, X_test_hc, _, _, y_test_hc, ts_test_hc = dp_homec.make_windows()
y_test_hc_kw = dp_homec.inverse_transform_power(y_test_hc)

lstm_hc = LSTMModel(window_size=24, n_features=8)
lstm_hc.load(os.path.join(BASE_DIR, "models", "lstm_homec.keras"))
y_pred_hc_kw = dp_homec.inverse_transform_power(lstm_hc.predict(X_test_hc))
metrics_hc   = lstm_hc.evaluate(y_test_hc_kw, y_pred_hc_kw)
print(f"HomeC готова. Метрики: {metrics_hc}")


# ------------------------------------------------------------------ #
#  Генерація підвибірок (секцій) з тестових даних                    #
# ------------------------------------------------------------------ #
def build_periods(timestamps, chunk_hours=336):
    """Розбиває масив timestamps на іменовані секції по chunk_hours годин."""
    periods = []
    ts = pd.to_datetime(timestamps)
    n  = len(ts)
    start = 0
    while start < n:
        end = min(start + chunk_hours, n)
        label = f"{ts[start].strftime('%d.%m')} – {ts[end-1].strftime('%d.%m.%Y')}"
        periods.append({"label": label, "start": start, "end": end})
        start += chunk_hours
    return periods

periods_uci   = build_periods(ts_test_uci,  chunk_hours=336)  # ~2 тижні
periods_homec = build_periods(ts_test_hc,   chunk_hours=168)  # ~1 тиждень

print(f"\nПідвибірок UCI   : {len(periods_uci)}")
print(f"Підвибірок HomeC : {len(periods_homec)}")
print("\nСервер готовий!")


# ------------------------------------------------------------------ #
#  Маршрути                                                           #
# ------------------------------------------------------------------ #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/datasets")
def api_datasets():
    return jsonify([
        {
            "id":          "uci",
            "name":        "UCI — Individual Household",
            "description": "Одне домогосподарство · Франція · 2006–2010 · Синтетична погода",
            "periods":     len(periods_uci),
            "metrics":     metrics_uci,
        },
        {
            "id":          "homec",
            "name":        "HomeC — Smart Home з погодою",
            "description": "Розумний будинок · США · 2016 · Реальна температура та вологість",
            "periods":     len(periods_homec),
            "metrics":     metrics_hc,
        },
    ])


@app.route("/api/periods")
def api_periods():
    dataset = request.args.get("dataset", "uci")
    periods = periods_uci if dataset == "uci" else periods_homec
    return jsonify([{"index": i, "label": p["label"]} for i, p in enumerate(periods)])


@app.route("/api/forecast")
def api_forecast():
    dataset  = request.args.get("dataset", "uci")
    period_i = int(request.args.get("period", 0))

    if dataset == "uci":
        periods, y_true, y_pred, ts, dp, metrics = (
            periods_uci, y_test_uci_kw, y_pred_uci_kw, ts_test_uci, dp_uci, metrics_uci
        )
    else:
        periods, y_true, y_pred, ts, dp, metrics = (
            periods_homec, y_test_hc_kw, y_pred_hc_kw, ts_test_hc, dp_homec, metrics_hc
        )

    period_i = max(0, min(period_i, len(periods) - 1))
    p        = periods[period_i]
    s, e     = p["start"], p["end"]

    labels      = [pd.Timestamp(t).strftime("%d.%m %H:%M") for t in ts[s:e]]
    actual_seg  = y_true[s:e]
    pred_seg    = y_pred[s:e]

    # Метрики для цієї секції
    seg_metrics = dp.scaler  # dummy — використовуємо LSTMModel.evaluate
    mae  = float(np.mean(np.abs(actual_seg - pred_seg)))
    rmse = float(np.sqrt(np.mean((actual_seg - pred_seg) ** 2)))
    mask = actual_seg > 0.3
    mape = float(np.mean(np.abs((actual_seg[mask] - pred_seg[mask]) / actual_seg[mask])) * 100) if mask.sum() > 0 else 0.0

    seg_metrics = {"MAE": round(mae,4), "RMSE": round(rmse,4), "MAPE": round(mape,2)}

    # Зберігаємо в БД
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_user_session(dataset, p["label"], now)
    save_prediction_session(dataset, p["label"], mae, rmse, mape, now)

    return jsonify({
        "labels":       labels,
        "actual":       [round(float(v), 3) for v in actual_seg],
        "forecast":     [round(float(v), 3) for v in pred_seg],
        "metrics":      seg_metrics,
        "period_label": p["label"],
    })


@app.route("/api/recommendations")
def api_recommendations():
    dataset  = request.args.get("dataset", "uci")
    period_i = int(request.args.get("period", 0))

    if dataset == "uci":
        periods, y_pred, ts = periods_uci, y_pred_uci_kw, ts_test_uci
    else:
        periods, y_pred, ts = periods_homec, y_pred_hc_kw, ts_test_hc

    period_i = max(0, min(period_i, len(periods) - 1))
    p        = periods[period_i]
    s, e     = p["start"], p["end"]

    PEAK_TARIFF  = 4.32
    NIGHT_TARIFF = 2.16
    hours        = np.arange(24)
    tariffs      = np.where((hours >= 7) & (hours < 23), PEAK_TARIFF, NIGHT_TARIFF)

    profile = np.zeros(24)
    counts  = np.zeros(24)
    for i in range(s, e):
        if i < len(ts):
            h = pd.Timestamp(ts[i]).hour
            profile[h] += y_pred[i]
            counts[h]  += 1
    counts[counts == 0] = 1
    profile /= counts

    cost_before = float(np.sum(profile * tariffs))
    optimized   = profile.copy()
    peak_mask   = (hours >= 18) & (hours < 22)
    night_mask  = (hours >= 23) | (hours < 6)
    shift       = optimized[peak_mask] * 0.25
    optimized[peak_mask]  -= shift
    optimized[night_mask] += shift.sum() / max(night_mask.sum(), 1)
    cost_after   = float(np.sum(optimized * tariffs))
    saving_pct   = round((cost_before - cost_after) / max(cost_before, 0.01) * 100, 1)
    saving_month = round((cost_before - cost_after) * 30, 0)

    recs = []
    peak_hours = [h for h in range(18, 22) if profile[h] > profile.mean()]
    if peak_hours:
        recs.append({
            "icon": "⚡",
            "title": "Вечірній пік навантаження",
            "text": f"З {peak_hours[0]}:00 до {peak_hours[-1]+1}:00 прогнозується підвищене споживання. "
                    "Рекомендується перенести роботу енергоємних приладів на нічний тариф (після 23:00)."
        })
    recs.append({
        "icon": "🌙",
        "title": "Нічний тариф (23:00 – 07:00)",
        "text": f"Тариф вночі вдвічі нижчий ({NIGHT_TARIFF} грн/кВт·год проти {PEAK_TARIFF} грн/кВт·год). "
                "Заряджайте електромобіль, запускайте тривалі цикли приладів у цей час."
    })
    if saving_pct > 0:
        recs.append({
            "icon": "💰",
            "title": f"Потенційна економія: ~{saving_month:.0f} грн/місяць",
            "text": f"При зміщенні 25% пікового споживання на нічні години "
                    f"орієнтовна економія складає {saving_pct}% від поточних витрат."
        })

    return jsonify({
        "profile_before": [round(float(v), 3) for v in profile],
        "profile_after":  [round(float(v), 3) for v in optimized],
        "hours":          list(range(24)),
        "cost_before":    round(cost_before, 2),
        "cost_after":     round(cost_after, 2),
        "saving_pct":     saving_pct,
        "saving_month":   saving_month,
        "recommendations": recs,
    })


@app.route("/api/history")
def api_history():
    return jsonify({
        "sessions": get_recent_sessions(10),
        "stats":    get_prediction_stats(),
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
