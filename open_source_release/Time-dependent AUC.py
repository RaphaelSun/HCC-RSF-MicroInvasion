from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
while not (ROOT / "common.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from common import REPEATS, build_model_frame, configure_matplotlib, fit_rsf, load_raw_data, split_xy, time_auc


configure_matplotlib()
raw = load_raw_data(Path(__file__).resolve().parent)
X, y, _, _ = build_model_frame(raw)
horizons = np.array([12.0, 24.0, 36.0, 48.0, 60.0])
results = []

for seed in range(REPEATS):
    X_train, X_test, y_train, y_test = split_xy(X, y)
    model = fit_rsf(X_train, y_train)
    results.append(time_auc(model, y_train, X_test, y_test, horizons))

scores = np.vstack(results)
medians = np.median(scores, axis=0)

fig, ax = plt.subplots(figsize=(10, 7))
bp = ax.boxplot(scores, patch_artist=True, labels=[f"{int(h/12)}-Year" for h in horizons], showfliers=False, widths=0.56)
for patch, color in zip(bp["boxes"], ["#EEF3FA", "#E8DBF5", "#E4D3F0", "#D8B7E6", "#C978CF"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.9)
    patch.set_edgecolor("#6257A0")
    patch.set_linewidth(2.2)
for item in bp["whiskers"] + bp["caps"]:
    item.set_color("#6257A0")
    item.set_linewidth(2.0)
for median in bp["medians"]:
    median.set_color("#D32F2F")
    median.set_linewidth(3.5)
for idx, median in enumerate(medians, start=1):
    y = scores[:, idx - 1]
    visible = y[y <= np.percentile(y, 75) + 1.5 * (np.percentile(y, 75) - np.percentile(y, 25))]
    anchor = float(visible.max() if len(visible) else y.max())
    ax.text(idx, anchor + 0.004, f"{median:.2f}", ha="center", va="bottom", fontsize=15, fontweight="bold", color="#28297A")
ax.set_title("Longitudinal Evaluation of Model AUC", fontsize=24, fontweight="bold", pad=20)
ax.set_ylabel("AUC Score", fontsize=20, fontweight="bold")
ax.set_xlabel("Time Horizon", fontsize=20, fontweight="bold", labelpad=14)
ax.tick_params(axis="both", labelsize=15)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
ax.grid(axis="y", linestyle="--", alpha=0.35, color="#B8BEC7")
for spine in ax.spines.values():
    spine.set_linewidth(2.2)
ax.set_ylim(0.72, 1.01)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("Time_Dependent_AUC_Boxplot.png"), bbox_inches="tight")

for horizon, column in zip(horizons, scores.T):
    print(f"{int(horizon/12)}-year AUC: mean={column.mean():.3f}, sd={column.std(ddof=1):.3f}")
