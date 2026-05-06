from pathlib import Path
import sys

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
while not (ROOT / "common.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from common import REPEATS, build_model_frame, confidence_interval, import_sksurv, load_raw_data, split_xy, to_surv


modules = import_sksurv()
raw = load_raw_data(Path(__file__).resolve().parent)
X, y, _, features = build_model_frame(raw)
scores = []

for seed in range(REPEATS):
    X_train, X_test, y_train, y_test = split_xy(X, y)
    model = make_pipeline(StandardScaler(), modules["CoxPHSurvivalAnalysis"]())
    model.fit(X_train, to_surv(y_train))
    scores.append(model.score(X_test, to_surv(y_test)))

mean_c, low_c, high_c = confidence_interval(scores)
print("Cox proportional hazards")
print(f"Features used ({len(features)}): {features}")
print(f"Mean validation C-index: {mean_c:.4f}")
print(f"95% CI: [{low_c:.4f}, {high_c:.4f}]")
