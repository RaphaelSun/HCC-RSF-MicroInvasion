from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
while not (ROOT / "common.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from common import build_model_frame, configure_matplotlib, fit_rsf, load_raw_data, select_representative_patients


configure_matplotlib()
raw = load_raw_data(Path(__file__).resolve().parent)
X, y, frame, _ = build_model_frame(raw)
model = fit_rsf(X, y)
survival_functions = model.predict_survival_function(X)
risk_scores = model.predict(X)
high_cases, low_cases, cutoff = select_representative_patients(frame.reset_index(drop=True), risk_scores)

fig, ax = plt.subplots(figsize=(12, 8.5))
reds = ["#7B241C", "#C0392B", "#E74C3C"]
blues = ["#154360", "#2E86C1", "#5DADE2"]

for color, (_, row) in zip(reds, high_cases.iterrows()):
    idx = int(row.name)
    curve = survival_functions[idx]
    t = np.concatenate(([0.0], curve.x))
    y_plot = (1.0 - np.concatenate(([1.0], curve.y))) * 100.0
    ax.step(t, y_plot, where="post", lw=3.1, color=color, label=f"High Risk | Score: {row['RiskScore']:.2f}")
    event_y = np.interp(row["Time"], t, y_plot)
    ax.scatter(row["Time"], event_y, s=110, marker="x", color=color, linewidth=3)
    ax.annotate(f"{row['Time']:.1f}m", xy=(row["Time"], event_y), xytext=(row["Time"] + 1.5, event_y - 5), fontsize=11, fontweight="bold", color=color)

for color, (_, row) in zip(blues, low_cases.iterrows()):
    idx = int(row.name)
    curve = survival_functions[idx]
    t = np.concatenate(([0.0], curve.x))
    y_plot = (1.0 - np.concatenate(([1.0], curve.y))) * 100.0
    ax.step(t, y_plot, where="post", lw=3.1, color=color, label=f"Low Risk | Score: {row['RiskScore']:.2f}")

ax.set_xlabel("Time (Months)", fontsize=18, fontweight="bold")
ax.set_ylabel("Cumulative Recurrence Probability (%)", fontsize=18, fontweight="bold")
ax.set_title("Personalized Cumulative Recurrence Trajectories", fontsize=22, fontweight="bold", pad=18)
ax.grid(linestyle="--", alpha=0.3, color="#C2CAD3")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.8)
ax.spines["bottom"].set_linewidth(1.8)
ax.tick_params(axis="both", labelsize=13, width=1.5, length=6)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
ax.set_xlim(0, 72)
ax.set_ylim(0, 105)
ax.legend(frameon=False, fontsize=11, loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("Personalized_Recurrence_Trajectories.png"), bbox_inches="tight")

print(f"Representative cutoff: {cutoff:.4f}")
