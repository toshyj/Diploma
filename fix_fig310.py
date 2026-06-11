import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = "figures"

def save_fig(name):
    plt.savefig(os.path.join(FIGURES_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Збережено: figures/{name}")

df_neurons = pd.read_csv("figures/table_neurons_ablation.csv", encoding="utf-8-sig")
print("Дані завантажено:")
print(df_neurons.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#FF7043", "#FF7043", "#4CAF50", "#FF7043", "#FF7043"]
sizes  = [100, 100, 220, 100, 100]
for i, row in df_neurons.iterrows():
    ax.scatter(row["Нейронів"], row["MAPE (%)"],
               color=colors[i], s=sizes[i], zorder=5,
               edgecolors="white", linewidths=0.8)
    ax.annotate(f'{row["MAPE (%)"]:.1f}%',
                (row["Нейронів"], row["MAPE (%)"]),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color="#555555")

ax.axvline(64, color="#4CAF50", linestyle="--",
           linewidth=1.5, label="Оптимум: 64 нейрони")
ax.set_xlabel("Кількість нейронів LSTM-шару")
ax.set_ylabel("MAPE (%)")
ax.set_title("Рисунок 3.10 — Діаграма розсіювання точності відносно кількості нейронів")
ax.legend()
ax.grid(True, alpha=0.3)
save_fig("fig_3_10_neurons_ablation.png")
print("Готово!")