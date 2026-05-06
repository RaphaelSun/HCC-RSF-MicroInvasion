from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
while not (ROOT / "common.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from common import build_model_frame, configure_matplotlib, encode_cnlc_early, fit_rsf, km_estimate, load_raw_data, locate_stage_columns, split_xy, style_axis, style_panel_title


configure_matplotlib()
raw = load_raw_data(Path(__file__).resolve().parent)
_, stage_col = locate_stage_columns(raw)
if stage_col is None:
    raise ValueError("CNLC staging column was not found.")
X, y, frame, _ = build_model_frame(raw)
X_train, X_test, y_train, y_test = split_xy(X, y)
model = fit_rsf(X_train, y_train)
frame["CNLC_Group"] = encode_cnlc_early(raw.loc[frame.index, stage_col])
frame["RiskScore"] = model.predict(X)
cutoff = float(np.median(model.predict(X_train)))
frame["RiskGroup"] = np.where(frame["RiskScore"] > cutoff, "High risk", "Low risk")

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

for label, color in [("Stage Ia", "#2C3E50"), ("Stage Ib", "#95A5A6"), ("Stage II", "#7F8C8D")]:
    subset = frame[frame["CNLC_Group"] == label]
    t, s, l, u = km_estimate(subset["Time"], subset["Event"])
    axes[0].step(t, s, where="post", lw=2.6, color=color, label=label)
    axes[0].fill_between(t, l, u, step="post", color=color, alpha=0.12)
style_panel_title(axes[0], "Traditional Clinical Staging (CNLC)")
style_axis(axes[0])
axes[0].legend(frameon=False)

stage_ia = frame[frame["CNLC_Group"] == "Stage Ia"]
for label, color in [("Low risk", "#2E86C1"), ("High risk", "#C0392B")]:
    subset = stage_ia[stage_ia["RiskGroup"] == label]
    t, s, l, u = km_estimate(subset["Time"], subset["Event"])
    axes[1].step(t, s, where="post", lw=2.6, color=color, label=label)
    axes[1].fill_between(t, l, u, step="post", color=color, alpha=0.12)
style_panel_title(axes[1], "RSF Stratification Within CNLC Stage Ia")
style_axis(axes[1])
axes[1].legend(frameon=False)

fig.tight_layout()
fig.savefig(Path(__file__).with_name("CNLC_EarlyStage_RiskStratification.png"), bbox_inches="tight")
