from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, auc

ROOT = Path(__file__).resolve().parent
while not (ROOT / "common.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from common import build_model_frame, configure_matplotlib, fit_rsf, km_estimate, load_raw_data, split_xy, style_axis, style_panel_title, to_surv


configure_matplotlib()
raw = load_raw_data(Path(__file__).resolve().parent)
X, y, frame, _ = build_model_frame(raw)
X_train, X_test, y_train, y_test = split_xy(X, y)
model = fit_rsf(X_train, y_train)
cutoff = float(np.median(model.predict(X_train)))

train_scores = model.predict(X_train)
test_scores = model.predict(X_test)

train_groups = {
    "Low risk": (y_train.loc[train_scores <= cutoff, "Time"], y_train.loc[train_scores <= cutoff, "Event"], "#2E86C1"),
    "High risk": (y_train.loc[train_scores > cutoff, "Time"], y_train.loc[train_scores > cutoff, "Event"], "#C0392B"),
}
test_groups = {
    "Low risk": (y_test.loc[test_scores <= cutoff, "Time"], y_test.loc[test_scores <= cutoff, "Event"], "#2E86C1"),
    "High risk": (y_test.loc[test_scores > cutoff, "Time"], y_test.loc[test_scores > cutoff, "Event"], "#C0392B"),
}

train_binary = ((y_train["Time"] <= 24) & (y_train["Event"])).astype(int)
test_binary = ((y_test["Time"] <= 24) & (y_test["Event"])).astype(int)
train_mask = ((y_train["Time"] <= 24) & (y_train["Event"])) | (y_train["Time"] > 24)
test_mask = ((y_test["Time"] <= 24) & (y_test["Event"])) | (y_test["Time"] > 24)
fpr_train, tpr_train, thr_train = roc_curve(train_binary[train_mask], train_scores[train_mask])
fpr_test, tpr_test, _ = roc_curve(test_binary[test_mask], test_scores[test_mask])
train_auc = auc(fpr_train, tpr_train)
test_auc = auc(fpr_test, tpr_test)
cut_idx = int(np.argmin(np.abs(thr_train - cutoff)))

fig = plt.figure(figsize=(16.5, 6.8))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.18, 1.18, 1.0], wspace=0.26)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]

for ax, groups, title in zip(axes[:2], [train_groups, test_groups], ["Training Cohort", "Validation Cohort"]):
    for label, (times, events, color) in groups.items():
        t, s, l, u = km_estimate(times.to_numpy(), events.to_numpy())
        ax.step(t, s, where="post", lw=2.6, color=color, label=label)
        ax.fill_between(t, l, u, step="post", color=color, alpha=0.12)
    style_panel_title(ax, title)
    style_axis(ax)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=11, loc="upper right")

ax = axes[2]
ax.plot(fpr_train, tpr_train, color="#2E86C1", lw=2.9, label=f"Train AUC = {train_auc:.2f}")
ax.plot(fpr_test, tpr_test, color="#C0392B", lw=2.9, label=f"Valid AUC = {test_auc:.2f}")
ax.scatter(fpr_train[cut_idx], tpr_train[cut_idx], s=90, color="#FF1E1E", label="Median Cutoff", zorder=6)
ax.plot([0, 1], [0, 1], linestyle="--", color="#A6A6A6", lw=1.8)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("1 - Specificity", fontsize=17, fontweight="bold")
ax.set_ylabel("Sensitivity", fontsize=17, fontweight="bold")
style_panel_title(ax, "Time-Dependent ROC (24m)")
ax.tick_params(axis="both", labelsize=13, width=1.5, length=6)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
ax.legend(frameon=False, fontsize=11, loc="lower right")
ax.grid(linestyle="--", alpha=0.28, color="#BFC7D0")
for spine in ax.spines.values():
    spine.set_linewidth(1.8)

fig.tight_layout()
fig.savefig(Path(__file__).with_name("Risk_Stratification_KM.png"), bbox_inches="tight")

print(f"Training cutoff: {cutoff:.4f}")
print(f"Training C-index: {model.score(X_train, to_surv(y_train)):.4f}")
print(f"Validation C-index: {model.score(X_test, to_surv(y_test)):.4f}")
