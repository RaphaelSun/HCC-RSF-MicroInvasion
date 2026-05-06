from pathlib import Path
import sys

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
while not (ROOT / "common.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from common import build_model_frame, configure_matplotlib, encode_bclc, fit_rsf, load_raw_data, locate_stage_columns


configure_matplotlib()
raw = load_raw_data(Path(__file__).resolve().parent)
stage_col, _ = locate_stage_columns(raw)
if stage_col is None:
    raise ValueError("BCLC staging column was not found.")
X, y, frame, _ = build_model_frame(raw)
model = fit_rsf(X, y)
frame["BCLC_Group"] = encode_bclc(raw.loc[frame.index, stage_col])
frame["RiskScore"] = model.predict(X)
plot_df = frame.dropna(subset=["BCLC_Group"]).copy()
order = [label for label in ["Stage 0", "Stage A", "Stage B", "Stage C", "Stage D"] if label in plot_df["BCLC_Group"].unique()]

fig, ax = plt.subplots(figsize=(10, 7))
sns.boxplot(data=plot_df, x="BCLC_Group", y="RiskScore", order=order, palette=sns.color_palette("Blues", len(order)), ax=ax, fliersize=0, width=0.52, linewidth=2)
medians = plot_df.groupby("BCLC_Group")["RiskScore"].median().reindex(order)
ax.plot(range(len(order)), medians, color="#C0392B", linestyle="--", linewidth=3, marker="o", markersize=8, zorder=5)
ax.set_title("Consistency: Risk Score VS BCLC Stage", fontsize=18, fontweight="bold", pad=18)
ax.set_xlabel("BCLC Stage", fontsize=15, fontweight="bold")
ax.set_ylabel("Risk Score", fontsize=15, fontweight="bold")
ax.tick_params(axis="both", labelsize=12, width=1.5, length=6)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
ax.grid(axis="y", linestyle="--", alpha=0.28, color="#BFC7D0")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.8)
ax.spines["bottom"].set_linewidth(1.8)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("BCLC_RiskScore_Boxplot.png"), bbox_inches="tight")
