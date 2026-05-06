from pathlib import Path
import sys

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
while not (ROOT / "common.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from common import build_model_frame, configure_matplotlib, encode_cnlc_groups, fit_rsf, load_raw_data, locate_stage_columns


configure_matplotlib()
raw = load_raw_data(Path(__file__).resolve().parent)
_, stage_col = locate_stage_columns(raw)
if stage_col is None:
    raise ValueError("CNLC staging column was not found.")
X, y, frame, _ = build_model_frame(raw)
model = fit_rsf(X, y)
frame["CNLC_Group"] = encode_cnlc_groups(raw.loc[frame.index, stage_col])
frame["RiskScore"] = model.predict(X)
plot_df = frame.dropna(subset=["CNLC_Group"]).copy()
order = [label for label in ["Stage I", "Stage II", "Stage III"] if label in plot_df["CNLC_Group"].unique()]

fig, ax = plt.subplots(figsize=(10, 7))
sns.boxplot(data=plot_df, x="CNLC_Group", y="RiskScore", order=order, palette=sns.color_palette("YlOrRd", len(order)), ax=ax, fliersize=0, width=0.52, linewidth=2)
medians = plot_df.groupby("CNLC_Group")["RiskScore"].median().reindex(order)
ax.plot(range(len(order)), medians, color="#C0392B", linestyle="--", linewidth=3, marker="o", markersize=8, zorder=5)
ax.set_title("Consistency: Risk Score VS CNLC Stages", fontsize=18, fontweight="bold", pad=18)
ax.set_xlabel("CNLC Stage", fontsize=15, fontweight="bold")
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
fig.savefig(Path(__file__).with_name("CNLC_RiskScore_Boxplot.png"), bbox_inches="tight")
