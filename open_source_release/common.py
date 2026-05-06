from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TEST_SIZE = 0.30
REPEATS = int(os.getenv("HCC_REPEAT_SPLITS", "100"))

MODEL_FEATURE_SPECS = [
    ("T", "T Stage", lambda col: col == "T"),
    ("Preoperative AFP (ng/ml)", "Preoperative AFP", lambda col: col == "Preoperative AFP (ng/ml)"),
    ("TB_status", "Tumor Budding (TB)", lambda col: col.upper().startswith("TB") and "NUMBER" not in col.upper()),
    ("MVI_status", "Microvascular Invasion (MVI)", lambda col: col.upper().startswith("MVI") and "DEGREE" not in col.upper()),
    ("Tumor_Volume", "Tumor Volume", lambda col: col == "Tumor_Volume"),
    ("TB number", "Tumor Budding Count", lambda col: col == "TB number"),
    ("Lymphocyte", "Lymphocyte", lambda col: col == "Lymphocyte"),
    ("Multiple tumors", "Multiple Tumors", lambda col: col == "Multiple tumors"),
    ("Preoperative AST (U/L)", "AST", lambda col: col == "Preoperative AST (U/L)"),
    ("Preoperative total bilirubin (umol/L)", "Total Bilirubin", lambda col: col == "Preoperative total bilirubin (umol/L)"),
    ("Preoperative CEA (ng/ml)", "Preoperative CEA", lambda col: col == "Preoperative CEA (ng/ml)"),
    ("Preoperative ALT (U/L)", "ALT", lambda col: col == "Preoperative ALT (U/L)"),
    ("Preoperative albumin (g/L)", "Albumin", lambda col: col == "Preoperative albumin (g/L)"),
    ("age (year)", "Age", lambda col: col == "age (year)"),
    ("FIB-4", "FIB-4 Index", lambda col: col == "FIB-4"),
    ("Neutrophils (X10^9/L)", "Neutrophils", lambda col: col == "Neutrophils (X10^9/L)"),
    ("Preoperative CA199 (U/ml)", "Preoperative CA199", lambda col: col == "Preoperative CA199 (U/ml)"),
    ("Prothrombin Time", "Prothrombin Time", lambda col: col == "Prothrombin Time"),
    ("HBV-DNA", "HBV-DNA", lambda col: "HBV-DNA" in col),
    ("Platelet", "Platelet Count", lambda col: col == "Platelet"),
    ("Gender (Male 1, Female 0)", "Gender", lambda col: col == "Gender (Male 1, Female 0)"),
    ("Degree of differentiation", "Differentiation Degree", lambda col: "Degree of differentiation" in col),
    ("Cirrhosis (Yes 1, No 0)", "Cirrhosis", lambda col: col == "Cirrhosis (Yes 1, No 0)"),
]


def resolve_dataset_path(start: Path | None = None) -> Path:
    roots = [start or Path.cwd()]
    roots.extend((start or Path.cwd()).parents)
    candidates = []
    for root in roots:
        candidates.extend(
            [
                root / "Dataset_Processed.xlsx",
                root / "Dataset_Processed.csv",
                root / ".." / "Dataset_Processed.xlsx",
                root / ".." / "Dataset_Processed.csv",
                root / ".." / ".." / "Dataset_Processed.xlsx",
                root / ".." / ".." / "Dataset_Processed.csv",
                root / ".." / ".." / ".." / "Dataset_Processed.xlsx",
                root / ".." / ".." / ".." / "Dataset_Processed.csv",
            ]
        )
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Dataset_Processed.xlsx or Dataset_Processed.csv was not found.")


def load_raw_data(base_dir: Path | None = None) -> pd.DataFrame:
    path = resolve_dataset_path(base_dir)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def force_numeric(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    for token in [">", "<"]:
        text = text.replace(token, "")
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_tumor_size(value):
    if pd.isna(value):
        return np.nan, np.nan
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return np.nan, np.nan
    text = re.sub(r"[^0-9.xX*]+", "", text).replace("X", "x").replace("*", "x")
    dims = []
    for chunk in text.split("x"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            dims.append(float(chunk))
        except ValueError:
            pass
    if not dims:
        return np.nan, np.nan
    max_diameter = max(dims)
    if len(dims) >= 3:
        volume = dims[0] * dims[1] * dims[2] * 0.52
    elif len(dims) == 2:
        volume = dims[0] * dims[1] * min(dims) * 0.52
    else:
        volume = max_diameter ** 3 * 0.52
    return max_diameter, volume


def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "Tumor Size (CM)" in data.columns:
        parsed = data["Tumor Size (CM)"].apply(parse_tumor_size)
        if "Tumor_Max_Diameter" not in data.columns:
            data["Tumor_Max_Diameter"] = parsed.apply(lambda x: x[0])
        if "Tumor_Volume" not in data.columns:
            data["Tumor_Volume"] = parsed.apply(lambda x: x[1])
    if "RFS_Composite_Event" in data.columns and "RFS_Composite_Time" in data.columns:
        data["Event"] = data["RFS_Composite_Event"].astype(bool)
        data["Time"] = pd.to_numeric(data["RFS_Composite_Time"], errors="coerce")
    else:
        data["Event"] = pd.to_numeric(data["RFS"], errors="coerce").fillna(0).astype(int).astype(bool)
        data["Time"] = pd.to_numeric(data["RFS.TIME"], errors="coerce")
    data = data[data["Time"].notna() & (data["Time"] > 0)].copy()
    return data


def locate_stage_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    bclc = next((col for col in df.columns if "BCLC" in str(col).upper()), None)
    cnlc = next((col for col in df.columns if "CNLC" in str(col).upper()), None)
    return bclc, cnlc


def resolve_features(df: pd.DataFrame, include_tb: bool = True, include_mvi: bool = True) -> list[str]:
    selected = []
    for feature, _, matcher in MODEL_FEATURE_SPECS:
        if not include_tb and feature == "TB_status":
            continue
        if not include_mvi and feature == "MVI_status":
            continue
        match = next((col for col in df.columns if matcher(str(col))), None)
        if match is not None:
            selected.append(match)
    return selected


def get_display_names(columns: list[str]) -> list[str]:
    display_names = []
    for col in columns:
        display = next((name for _, name, matcher in MODEL_FEATURE_SPECS if matcher(str(col))), col)
        display_names.append(display)
    return display_names


def build_model_frame(
    df: pd.DataFrame,
    include_tb: bool = True,
    include_mvi: bool = True,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    data = standardize_data(df)
    selected = features or resolve_features(data, include_tb=include_tb, include_mvi=include_mvi)
    frame = data[selected + ["Event", "Time"]].copy()
    for col in selected:
        frame[col] = frame[col].apply(force_numeric)
        frame[col] = frame[col].fillna(frame[col].median())
        frame[col] = frame[col].fillna(0.0)
    frame = frame.dropna(subset=["Time"])
    X = frame[selected].copy()
    y = frame[["Event", "Time"]].copy()
    return X, y, frame, selected


def import_sksurv():
    try:
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.metrics import cumulative_dynamic_auc
        from sksurv.svm import FastKernelSurvivalSVM
        from sksurv.util import Surv
    except ImportError as exc:
        raise ImportError("This script requires scikit-survival. Install `scikit-survival` before running.") from exc
    return {
        "RandomSurvivalForest": RandomSurvivalForest,
        "GradientBoostingSurvivalAnalysis": GradientBoostingSurvivalAnalysis,
        "CoxPHSurvivalAnalysis": CoxPHSurvivalAnalysis,
        "FastKernelSurvivalSVM": FastKernelSurvivalSVM,
        "Surv": Surv,
        "cumulative_dynamic_auc": cumulative_dynamic_auc,
    }


def to_surv(y: pd.DataFrame):
    surv = import_sksurv()["Surv"]
    return surv.from_arrays(event=y["Event"].astype(bool).to_numpy(), time=y["Time"].to_numpy())


def split_xy(X: pd.DataFrame, y: pd.DataFrame, seed: int | None = None):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y["Event"].astype(int),
    )


def build_rsf_model(seed: int | None = None):
    rsf = import_sksurv()["RandomSurvivalForest"]
    if seed is None:
        return rsf()
    return rsf(random_state=seed)


def fit_rsf(X_train: pd.DataFrame, y_train: pd.DataFrame, seed: int | None = None):
    model = build_rsf_model(seed)
    model.fit(X_train, to_surv(y_train))
    return model


def confidence_interval(values) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    margin = 1.96 * sd / np.sqrt(arr.size) if arr.size > 1 else 0.0
    return mean, mean - margin, mean + margin


def km_estimate(times, events):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    order = np.argsort(times)
    times = times[order]
    events = events[order]
    unique_times = np.unique(times)
    curve_t = [0.0]
    curve_s = [1.0]
    curve_l = [1.0]
    curve_u = [1.0]
    survival = 1.0
    var_term = 0.0
    for t in unique_times:
        at_risk = np.sum(times >= t)
        occurred = np.sum((times == t) & events)
        if at_risk <= 0:
            break
        if occurred > 0 and occurred < at_risk:
            survival *= 1.0 - occurred / at_risk
            var_term += occurred / (at_risk * (at_risk - occurred))
        elif occurred >= at_risk:
            survival = 0.0
        se = survival * np.sqrt(max(var_term, 0.0))
        curve_t.append(float(t))
        curve_s.append(float(survival))
        curve_l.append(float(np.clip(survival - 1.96 * se, 0.0, 1.0)))
        curve_u.append(float(np.clip(survival + 1.96 * se, 0.0, 1.0)))
    return np.asarray(curve_t), np.asarray(curve_s), np.asarray(curve_l), np.asarray(curve_u)


def style_axis(ax, xlabel: str = "Time (Months)", ylabel: str = "Recurrence-Free Survival"):
    ax.set_xlabel(xlabel, fontsize=17, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=17, fontweight="bold")
    ax.tick_params(axis="both", labelsize=13, width=1.5, length=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(linestyle="--", alpha=0.28, color="#BFC7D0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)


def style_panel_title(ax, title: str):
    ax.set_title(title, fontsize=18, fontweight="bold", pad=14)


def encode_bclc(series: pd.Series) -> pd.Series:
    mapped = []
    for value in series:
        text = str(value).strip().upper()
        if text in {"0", "0.0"}:
            mapped.append("Stage 0")
        elif "A" in text or text in {"1", "1.0"}:
            mapped.append("Stage A")
        elif "B" in text or text in {"2", "2.0"}:
            mapped.append("Stage B")
        elif "C" in text or text in {"3", "3.0"}:
            mapped.append("Stage C")
        elif "D" in text or text in {"4", "4.0"}:
            mapped.append("Stage D")
        else:
            mapped.append(np.nan)
    return pd.Series(mapped, index=series.index)


def encode_cnlc_groups(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = []
    for value in numeric:
        if value in {1, 2}:
            mapped.append("Stage I")
        elif value in {3, 4}:
            mapped.append("Stage II")
        elif value in {5, 6}:
            mapped.append("Stage III")
        else:
            mapped.append(np.nan)
    return pd.Series(mapped, index=series.index)


def encode_cnlc_early(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = []
    for value in numeric:
        if value == 1:
            mapped.append("Stage Ia")
        elif value == 2:
            mapped.append("Stage Ib")
        elif value in {3, 4}:
            mapped.append("Stage II")
        else:
            mapped.append(np.nan)
    return pd.Series(mapped, index=series.index)


def time_auc(model, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, horizons):
    modules = import_sksurv()
    train_surv = to_surv(y_train)
    test_surv = to_surv(y_test)
    risk_scores = model.predict(X_test)
    auc_values, _ = modules["cumulative_dynamic_auc"](train_surv, test_surv, risk_scores, horizons)
    return np.asarray(auc_values, dtype=float)


def select_representative_patients(frame: pd.DataFrame, risk_scores, high_n: int = 3, low_n: int = 3):
    eval_df = frame.copy()
    eval_df["RiskScore"] = np.asarray(risk_scores)
    cutoff = np.median(eval_df["RiskScore"])
    high_pool = eval_df[(eval_df["RiskScore"] > cutoff) & (eval_df["Event"])]
    low_pool = eval_df[(eval_df["RiskScore"] <= cutoff) & (~eval_df["Event"])]
    high = high_pool.nlargest(high_n, "RiskScore")
    low = low_pool.nsmallest(low_n, "RiskScore")
    return high, low, cutoff


def prepare_output_path(script_path: Path, stem: str, suffix: str = ".png") -> Path:
    out_dir = script_path.resolve().parent
    return out_dir / f"{stem}{suffix}"


def configure_matplotlib():
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"
