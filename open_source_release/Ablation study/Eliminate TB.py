from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
while not (ROOT / "common.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from common import REPEATS, build_model_frame, confidence_interval, fit_rsf, load_raw_data, split_xy, to_surv


raw = load_raw_data(Path(__file__).resolve().parent)
X, y, _, features = build_model_frame(raw, include_tb=False, include_mvi=True)
scores = []

for seed in range(REPEATS):
    X_train, X_test, y_train, y_test = split_xy(X, y)
    model = fit_rsf(X_train, y_train)
    scores.append(model.score(X_test, to_surv(y_test)))

mean_c, low_c, high_c = confidence_interval(scores)
print("Ablation model excluding tumor budding")
print(f"Features used ({len(features)}): {features}")
print(f"Mean validation C-index: {mean_c:.4f}")
print(f"95% CI: [{low_c:.4f}, {high_c:.4f}]")
