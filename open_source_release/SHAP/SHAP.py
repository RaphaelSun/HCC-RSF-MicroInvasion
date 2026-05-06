from pathlib import Path
import sys

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
while not (ROOT / "common.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from common import build_model_frame, configure_matplotlib, fit_rsf, get_display_names, load_raw_data, to_surv


configure_matplotlib()
raw = load_raw_data(Path(__file__).resolve().parent)
X, y, _, features = build_model_frame(raw)
model = fit_rsf(X, y)
display_names = get_display_names(features)

importance = None
shap_values = None
try:
    import shap

    background = shap.sample(X, min(80, len(X)))
    explain_data = X.iloc[: min(180, len(X))].copy()
    explainer = shap.Explainer(lambda values: model.predict(pd.DataFrame(values, columns=X.columns)), background)
    shap_values = explainer(explain_data)
    importance = np.abs(shap_values.values).mean(axis=0)
except Exception:
    perm = permutation_importance(model, X, to_surv(y), n_repeats=20, scoring=None)
    importance = perm.importances_mean

importance_df = pd.DataFrame(
    {"Feature": X.columns, "Display": display_names, "Importance": importance}
).sort_values("Importance", ascending=False)

fig = plt.figure(figsize=(14, 9))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1.0], wspace=0.34)

if shap_values is not None:
    ax_left = fig.add_subplot(gs[0, 0])
    plt.sca(ax_left)
    import shap

    shap.summary_plot(
        shap_values.values,
        features=explain_data.rename(columns=dict(zip(features, display_names))),
        feature_names=display_names,
        show=False,
        max_display=len(display_names),
        color_bar=True,
    )
    fig = plt.gcf()
    ax_left = fig.axes[0]
    ax_left.axvline(0, color="#9E9E9E", lw=1.2)
    ax_left.set_xlabel("SHAP value", fontsize=16, fontweight="bold")
    ax_left.set_ylabel("")
    ax_left.tick_params(axis="both", labelsize=11)
    for label in ax_left.get_xticklabels() + ax_left.get_yticklabels():
        label.set_fontweight("bold")
    ax_left.text(-0.14, 1.02, "A", transform=ax_left.transAxes, fontsize=20, fontweight="bold")
    if len(fig.axes) > 1:
        colorbar_ax = fig.axes[-1]
        colorbar_ax.set_ylabel("Feature value", fontsize=15, fontweight="bold")
        colorbar_ax.tick_params(labelsize=0, length=0)
        colorbar_ax.text(2.6, 1.01, "High", transform=colorbar_ax.transAxes, fontsize=18, fontweight="bold")
        colorbar_ax.text(2.6, -0.01, "Low", transform=colorbar_ax.transAxes, fontsize=18, fontweight="bold")
else:
    ax_left = fig.add_subplot(gs[0, 0])
    top_left = importance_df.iloc[:20].iloc[::-1]
    ax_left.barh(top_left["Display"], top_left["Importance"], color="#4A90C2")
    ax_left.set_xlabel("Importance", fontsize=16, fontweight="bold")
    ax_left.tick_params(axis="both", labelsize=11)
    for label in ax_left.get_xticklabels() + ax_left.get_yticklabels():
        label.set_fontweight("bold")
    ax_left.text(-0.14, 1.02, "A", transform=ax_left.transAxes, fontsize=20, fontweight="bold")

ax_right = fig.add_subplot(gs[0, 1])
top_right = importance_df.iloc[:23].iloc[::-1]
bars = ax_right.barh(top_right["Display"], top_right["Importance"], color="#5E9DCE", edgecolor="#4E6C8B", linewidth=1.0)
ax_right.set_title("Feature Importance", fontsize=14, fontweight="bold", pad=8)
ax_right.set_xlabel("Mean |SHAP value|", fontsize=15, fontweight="bold")
ax_right.set_ylabel("")
ax_right.tick_params(axis="both", labelsize=11)
for label in ax_right.get_xticklabels() + ax_right.get_yticklabels():
    label.set_fontweight("bold")
ax_right.grid(axis="x", linestyle="--", alpha=0.22, color="#C8D2DC")
for value, bar in zip(top_right["Importance"], bars):
    ax_right.text(value + 0.03, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=10.5, fontweight="bold", color="#333333")
ax_right.text(-0.16, 1.02, "B", transform=ax_right.transAxes, fontsize=20, fontweight="bold")
fig.tight_layout()
fig.savefig(Path(__file__).with_name("SHAP_Combined.png"), bbox_inches="tight")

print(importance_df[["Display", "Importance"]].head(23).to_string(index=False))
