from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
while not (ROOT / "common.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from common import REPEATS, build_model_frame, confidence_interval, load_raw_data


try:
    import torch
    import torchtuples as tt
    from pycox.evaluation import EvalSurv
    from pycox.models import CoxPH
except ImportError as exc:
    raise ImportError("This script requires torch, torchtuples, and pycox.") from exc


raw = load_raw_data(Path(__file__).resolve().parent)
X, y, _, features = build_model_frame(raw)
X_all = X.to_numpy(dtype="float32")
durations = y["Time"].to_numpy(dtype="float32")
events = y["Event"].astype("int32").to_numpy()
scores = []

for seed in range(REPEATS):
    X_train, X_test, dur_train, dur_test, evt_train, evt_test = train_test_split(
        X_all,
        durations,
        events,
        test_size=0.30,
        stratify=events,
    )
    X_train, X_val, dur_train, dur_val, evt_train, evt_val = train_test_split(
        X_train,
        dur_train,
        evt_train,
        test_size=0.20,
        stratify=evt_train,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    net = tt.practical.MLPVanilla(X_train.shape[1], [32, 32], 1, batch_norm=True, dropout=0.1)
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.01)
    model.fit(
        X_train,
        (dur_train, evt_train),
        batch_size=128,
        epochs=100,
        callbacks=[tt.cb.EarlyStopping(patience=10)],
        val_data=(X_val, (dur_val, evt_val)),
        verbose=False,
    )
    model.compute_baseline_hazards()
    surv = model.predict_surv_df(X_test)
    scores.append(EvalSurv(surv, dur_test, evt_test, censor_surv="km").concordance_td())

mean_c, low_c, high_c = confidence_interval(scores)
print("DeepSurv")
print(f"Features used ({len(features)}): {features}")
print(f"Mean validation C-index: {mean_c:.4f}")
print(f"95% CI: [{low_c:.4f}, {high_c:.4f}]")
