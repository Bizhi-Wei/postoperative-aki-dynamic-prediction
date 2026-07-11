"""v13 model recalibration and measurement-intensity sensitivity analyses.

This script performs two reviewer-facing robustness analyses without changing
the primary v5-v12 model outputs.

1. Calibration update/recalibration
   - selected model at each landmark only: 0h XGBoost, 6h XGBoost, 24h LR;
   - subject-level v5 train/test split is preserved;
   - recalibrators are learned from grouped out-of-fold predictions in the
     development set, then applied once to the held-out test set;
   - no test-set refitting is performed.

2. Measurement-intensity sensitivity
   - current full predictor set;
   - remove per-feature lab/vital count predictors;
   - remove per-feature counts and imputation missingness indicators;
   - add aggregate lab/vital measurement-intensity counts from v11;
   - aggregate measurement-intensity-only diagnostic model.

No SHAP, manuscript writing, or external validation is performed here.
"""

from __future__ import annotations

from pathlib import Path
import re
import textwrap
import warnings
import zlib

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from develop_models_v5 import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    LANDMARKS,
    METADATA,
    OUTCOME,
    RANDOM_STATE,
    choose_grouped_split,
    load_data,
)


warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = SCRIPT_DIR.parent
V11_DIR = PROJECT_ROOT / "outputs" / "modeling_v11_missingness_measurement_intensity"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v13_recalibration_measurement_intensity"

SELECTED_MODEL = {
    0: "XGBoost",
    6: "XGBoost",
    24: "Logistic Regression",
}

BOOTSTRAPS = 500
OOF_FOLDS = 5

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}


def bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).astype(bool)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def predictor_columns(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if c not in {*METADATA, OUTCOME}]


def is_feature_count_column(column: str) -> bool:
    return bool(re.match(r"^(lab|vital)_(pre24h|0_\d+h)_.+_count$", column))


def aggregate_intensity_columns(table: pd.DataFrame) -> list[str]:
    return [
        c
        for c in table.columns
        if c != "stay_id"
        and c != OUTCOME
        and (
            c.endswith("_total_count")
            or c.endswith("_distinct_features")
            or c in {"all_measurement_total_count", "all_measurement_distinct_features"}
        )
    ]


def load_measurement_counts(landmark: int) -> pd.DataFrame:
    path = V11_DIR / f"audit_v11_measurement_counts_{landmark}h.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, low_memory=False)
    cols = ["stay_id", *aggregate_intensity_columns(table)]
    out = table[cols].copy()
    for col in cols:
        if col != "stay_id":
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def add_measurement_counts(data: pd.DataFrame, landmark: int) -> pd.DataFrame:
    counts = load_measurement_counts(landmark)
    merged = data.merge(counts, on="stay_id", how="left", validate="one_to_one")
    for col in aggregate_intensity_columns(merged):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)
    return merged


def identify_types_for_columns(data: pd.DataFrame, predictors: list[str]) -> tuple[list[str], list[str], list[str]]:
    categorical = [c for c in CATEGORICAL_COLUMNS if c in predictors]
    binary: list[str] = []
    continuous: list[str] = []
    for column in predictors:
        if column in categorical:
            continue
        series = data[column]
        if pd.api.types.is_bool_dtype(series):
            binary.append(column)
            continue
        observed = pd.to_numeric(series, errors="coerce").dropna()
        unique = set(observed.unique())
        if unique and unique.issubset({0, 1}):
            binary.append(column)
        elif pd.api.types.is_numeric_dtype(series) or len(observed) > 0:
            continuous.append(column)
        else:
            categorical.append(column)
    return continuous, binary, categorical


def make_preprocessor_custom(
    continuous: list[str],
    binary: list[str],
    categorical: list[str],
    *,
    scale: bool,
    add_missing_indicators: bool,
) -> ColumnTransformer:
    continuous_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=add_missing_indicators)),
    ]
    if scale:
        continuous_steps.append(("scaler", StandardScaler()))
    binary_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=add_missing_indicators)),
    ]
    categorical_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
    return ColumnTransformer(
        [
            ("continuous", Pipeline(continuous_steps), continuous),
            ("binary", Pipeline(binary_steps), binary),
            ("categorical", Pipeline(categorical_steps), categorical),
        ],
        remainder="drop",
    )


def selected_pipeline(
    model_name: str,
    continuous: list[str],
    binary: list[str],
    categorical: list[str],
    *,
    add_missing_indicators: bool = True,
) -> Pipeline:
    if model_name == "Logistic Regression":
        return Pipeline(
            [
                (
                    "preprocess",
                    make_preprocessor_custom(
                        continuous,
                        binary,
                        categorical,
                        scale=True,
                        add_missing_indicators=add_missing_indicators,
                    ),
                ),
                ("model", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE)),
            ]
        )
    if model_name == "XGBoost":
        return Pipeline(
            [
                (
                    "preprocess",
                    make_preprocessor_custom(
                        continuous,
                        binary,
                        categorical,
                        scale=False,
                        add_missing_indicators=add_missing_indicators,
                    ),
                ),
                (
                    "model",
                    xgb.XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.03,
                        max_depth=4,
                        min_child_weight=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if model_name == "Random Forest":
        return Pipeline(
            [
                (
                    "preprocess",
                    make_preprocessor_custom(
                        continuous,
                        binary,
                        categorical,
                        scale=False,
                        add_missing_indicators=add_missing_indicators,
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model: {model_name}")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)
        model.fit(logit(p).reshape(-1, 1), y)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except Exception:
        return np.nan, np.nan


def intercept_only_alpha(y: np.ndarray, p: np.ndarray) -> float:
    x = logit(p)
    y = np.asarray(y, dtype=float)
    try:
        from scipy.optimize import minimize_scalar

        def nll(alpha: float) -> float:
            eta = alpha + x
            return float(np.sum(np.logaddexp(0, eta) - y * eta))

        result = minimize_scalar(nll, bounds=(-5, 5), method="bounded")
        return float(result.x)
    except Exception:
        observed = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        expected = np.clip(np.mean(p), 1e-6, 1 - 1e-6)
        return float(np.log(observed / (1 - observed)) - np.log(expected / (1 - expected)))


def fit_recalibrators(y_cal: np.ndarray, p_cal: np.ndarray) -> dict[str, object]:
    p_cal = np.clip(np.asarray(p_cal, dtype=float), 1e-6, 1 - 1e-6)
    alpha = intercept_only_alpha(y_cal, p_cal)

    logistic = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)
    logistic.fit(logit(p_cal).reshape(-1, 1), y_cal)

    isotonic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    isotonic.fit(p_cal, y_cal)

    return {
        "intercept_update": alpha,
        "logistic_recalibration": logistic,
        "isotonic_recalibration": isotonic,
    }


def apply_recalibrator(method: str, recalibrator: object, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if method == "none":
        return p
    if method == "intercept_update":
        return np.clip(expit(float(recalibrator) + logit(p)), 1e-6, 1 - 1e-6)
    if method == "logistic_recalibration":
        model = recalibrator
        return np.clip(model.predict_proba(logit(p).reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
    if method == "isotonic_recalibration":
        model = recalibrator
        return np.clip(model.predict(p), 1e-6, 1 - 1e-6)
    raise ValueError(method)


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    observed = float(y.mean())
    expected = float(p.mean())
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "auroc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "auprc": average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "brier_score": brier_score_loss(y, p),
        "observed_risk": observed,
        "mean_predicted_risk": expected,
        "observed_expected_ratio": observed / expected if expected > 0 else np.nan,
        "absolute_calibration_error": abs(observed - expected),
        "calibration_in_large": logit(np.array([observed]))[0] - logit(np.array([expected]))[0]
        if 0 < observed < 1 and 0 < expected < 1
        else np.nan,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def subject_bootstrap_indices(subjects: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(subjects)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    lookup = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    return np.concatenate([lookup[subject] for subject in sampled])


def paired_delta_ci(
    y: np.ndarray,
    p_reference: np.ndarray,
    p_variant: np.ndarray,
    subjects: np.ndarray,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    metrics = ["auroc", "auprc", "brier_score", "absolute_calibration_error"]
    values = {m: [] for m in metrics}
    attempts = 0
    while len(values["brier_score"]) < BOOTSTRAPS and attempts < BOOTSTRAPS * 3:
        attempts += 1
        idx = subject_bootstrap_indices(subjects, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        ref = calibration_metrics(y[idx], p_reference[idx])
        var = calibration_metrics(y[idx], p_variant[idx])
        for metric in metrics:
            values[metric].append(var[metric] - ref[metric])
    out: dict[str, float] = {"bootstrap_successful_n": len(values["brier_score"])}
    for metric, vals in values.items():
        out[f"delta_{metric}_ci_lower"] = float(np.quantile(vals, 0.025)) if vals else np.nan
        out[f"delta_{metric}_ci_upper"] = float(np.quantile(vals, 0.975)) if vals else np.nan
    return out


def grouped_oof_predictions(
    data: pd.DataFrame,
    train_mask: pd.Series,
    predictors: list[str],
    model_name: str,
    *,
    add_missing_indicators: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = data.loc[train_mask].copy()
    test = data.loc[~train_mask].copy()
    y_train = train[OUTCOME].to_numpy(dtype=int)
    y_test = test[OUTCOME].to_numpy(dtype=int)
    groups = train["subject_id"].to_numpy(dtype=int)
    oof = np.full(len(train), np.nan)

    continuous, binary, categorical = identify_types_for_columns(train[predictors], predictors)
    splitter = GroupKFold(n_splits=min(OOF_FOLDS, len(np.unique(groups))))
    for fold, (fit_idx, cal_idx) in enumerate(splitter.split(train, y_train, groups), start=1):
        pipe = selected_pipeline(
            model_name,
            continuous,
            binary,
            categorical,
            add_missing_indicators=add_missing_indicators,
        )
        pipe.fit(train.iloc[fit_idx][predictors], y_train[fit_idx])
        oof[cal_idx] = pipe.predict_proba(train.iloc[cal_idx][predictors])[:, 1]

    if np.isnan(oof).any():
        raise ValueError("OOF predictions contain missing values")

    final_pipe = selected_pipeline(
        model_name,
        continuous,
        binary,
        categorical,
        add_missing_indicators=add_missing_indicators,
    )
    final_pipe.fit(train[predictors], y_train)
    test_prob = final_pipe.predict_proba(test[predictors])[:, 1]
    return y_train, oof, y_test, test_prob, test["subject_id"].to_numpy(dtype=int)


def train_predict_variant(
    data: pd.DataFrame,
    train_mask: pd.Series,
    predictors: list[str],
    model_name: str,
    *,
    add_missing_indicators: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = data.loc[train_mask].copy()
    test = data.loc[~train_mask].copy()
    y_test = test[OUTCOME].to_numpy(dtype=int)
    subjects = test["subject_id"].to_numpy(dtype=int)
    y_train = train[OUTCOME].to_numpy(dtype=int)
    continuous, binary, categorical = identify_types_for_columns(train[predictors], predictors)
    pipe = selected_pipeline(
        model_name,
        continuous,
        binary,
        categorical,
        add_missing_indicators=add_missing_indicators,
    )
    pipe.fit(train[predictors], y_train)
    p = pipe.predict_proba(test[predictors])[:, 1]
    return y_test, p, subjects


def variant_predictor_sets(data: pd.DataFrame, landmark: int) -> dict[str, tuple[list[str], bool]]:
    all_predictors = predictor_columns(data)
    aggregate_counts = aggregate_intensity_columns(data)
    original_predictors = [c for c in all_predictors if c not in aggregate_counts]
    per_feature_counts = [c for c in original_predictors if is_feature_count_column(c)]
    no_count_predictors = [c for c in original_predictors if c not in per_feature_counts]
    non_count_with_aggregate = list(dict.fromkeys([*no_count_predictors, *aggregate_counts]))
    return {
        "full_current": (original_predictors, True),
        "no_feature_count_predictors": (no_count_predictors, True),
        "no_count_no_imputation_indicators": (no_count_predictors, False),
        "no_count_plus_aggregate_intensity": (non_count_with_aggregate, True),
        "aggregate_intensity_only": (aggregate_counts, False),
    }


def calibration_bin_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    table = pd.DataFrame({"y": y, "p": p})
    try:
        table["bin"] = pd.qcut(table["p"], q=bins, duplicates="drop")
    except ValueError:
        table["bin"] = pd.cut(table["p"], bins=min(bins, max(2, table["p"].nunique())), include_lowest=True)
    out = (
        table.groupby("bin", observed=True)
        .agg(
            n=("y", "size"),
            observed_risk=("y", "mean"),
            mean_predicted_risk=("p", "mean"),
            predicted_min=("p", "min"),
            predicted_max=("p", "max"),
        )
        .reset_index(drop=True)
    )
    out["bin_id"] = np.arange(1, len(out) + 1)
    return out


def make_recalibration_figures(predictions: pd.DataFrame, perf: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    for landmark in LANDMARKS:
        panel = predictions.loc[predictions["landmark_hours"].eq(landmark)].copy()
        y = panel["y_true"].to_numpy(dtype=int)
        methods = [
            ("none", "Uncalibrated", "#4c78a8"),
            ("intercept_update", "Intercept update", "#f58518"),
            ("logistic_recalibration", "Logistic recalibration", "#54a24b"),
            ("isotonic_recalibration", "Isotonic recalibration", "#b279a2"),
        ]
        fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=180)
        for method, label, color in methods:
            bins = calibration_bin_table(y, panel[f"prob_{method}"].to_numpy(dtype=float), bins=10)
            ax.plot(
                bins["mean_predicted_risk"],
                bins["observed_risk"],
                marker="o",
                linewidth=1.4,
                label=label,
                color=color,
            )
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted risk")
        ax.set_ylabel("Observed risk")
        ax.set_title(f"{landmark} h selected-model calibration update")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"figure_v13_recalibration_curve_{landmark}h.png", dpi=300)
        plt.close(fig)

    selected = perf.copy()
    base = selected.loc[selected["recalibration_method"].eq("none"), ["landmark_hours", "brier_score"]].rename(
        columns={"brier_score": "base_brier"}
    )
    plot = selected.merge(base, on="landmark_hours", how="left")
    plot["delta_brier"] = plot["brier_score"] - plot["base_brier"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=180)
    methods = [m for m in plot["recalibration_method"].unique() if m != "none"]
    x = np.arange(len(LANDMARKS))
    width = 0.22
    colors = {"intercept_update": "#f58518", "logistic_recalibration": "#54a24b", "isotonic_recalibration": "#b279a2"}
    for i, method in enumerate(methods):
        sub = plot.loc[plot["recalibration_method"].eq(method)].sort_values("landmark_hours")
        ax.bar(x + (i - 1) * width, sub["delta_brier"], width=width, label=method.replace("_", " "), color=colors.get(method))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels=[f"{lm} h" for lm in LANDMARKS])
    ax.set_ylabel("Δ Brier score vs uncalibrated")
    ax.set_title("Calibration update effect on Brier score")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_v13_recalibration_brier_delta.png", dpi=300)
    plt.close(fig)


def make_measurement_figures(perf: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    base = perf.loc[perf["variant"].eq("full_current"), ["landmark_hours", "auroc", "brier_score"]].rename(
        columns={"auroc": "base_auroc", "brier_score": "base_brier"}
    )
    plot = perf.merge(base, on="landmark_hours", how="left")
    plot = plot.loc[~plot["variant"].eq("full_current")].copy()
    plot["delta_auroc"] = plot["auroc"] - plot["base_auroc"]
    plot["delta_brier"] = plot["brier_score"] - plot["base_brier"]
    labels = {
        "no_feature_count_predictors": "Remove per-feature counts",
        "no_count_no_imputation_indicators": "Remove counts + missingness indicators",
        "no_count_plus_aggregate_intensity": "Add aggregate intensity",
        "aggregate_intensity_only": "Intensity only",
    }

    for metric, ylabel, fname in [
        ("delta_auroc", "Δ AUROC vs current full model", "figure_v13_measurement_intensity_auroc_delta.png"),
        ("delta_brier", "Δ Brier score vs current full model", "figure_v13_measurement_intensity_brier_delta.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=180)
        x = np.arange(len(LANDMARKS))
        variants = [v for v in labels if v in set(plot["variant"])]
        width = 0.18
        for i, variant in enumerate(variants):
            sub = plot.loc[plot["variant"].eq(variant)].sort_values("landmark_hours")
            ax.bar(x + (i - 1.5) * width, sub[metric], width=width, label=labels[variant])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, labels=[f"{lm} h" for lm in LANDMARKS])
        ax.set_ylabel(ylabel)
        ax.set_title("Measurement-intensity sensitivity")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / fname, dpi=300)
        plt.close(fig)


def run_recalibration(train_subjects: set[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    perf_rows = []
    delta_rows = []
    prediction_tables = []
    for landmark in LANDMARKS:
        data = load_data(landmark)
        train_mask = data["subject_id"].astype(int).isin(train_subjects)
        model_name = SELECTED_MODEL[landmark]
        predictors = predictor_columns(data)
        y_train, oof_prob, y_test, test_prob, test_subjects = grouped_oof_predictions(
            data,
            train_mask,
            predictors,
            model_name,
            add_missing_indicators=True,
        )
        recalibrators = fit_recalibrators(y_train, oof_prob)
        method_probs = {"none": np.clip(test_prob, 1e-6, 1 - 1e-6)}
        for method, recalibrator in recalibrators.items():
            method_probs[method] = apply_recalibrator(method, recalibrator, test_prob)

        pred_table = data.loc[~train_mask, ["subject_id", "hadm_id", "stay_id", "landmark_hours"]].copy()
        pred_table["y_true"] = y_test
        for method, p in method_probs.items():
            pred_table[f"prob_{method}"] = p
        prediction_tables.append(pred_table)
        pred_table.to_csv(OUTPUT_DIR / f"model_v13_recalibrated_predictions_{landmark}h.csv", index=False)

        for method, p in method_probs.items():
            row = {
                "landmark_hours": landmark,
                "selected_model": model_name,
                "recalibration_method": method,
                "train_n_for_oof_recalibration": len(y_train),
                "test_n": len(y_test),
                "test_event_n": int(y_test.sum()),
                "test_event_rate": float(y_test.mean()),
            }
            row.update(calibration_metrics(y_test, p))
            perf_rows.append(row)
            if method != "none":
                ref = method_probs["none"]
                delta = {
                    "landmark_hours": landmark,
                    "selected_model": model_name,
                    "recalibration_method": method,
                    "delta_auroc": calibration_metrics(y_test, p)["auroc"] - calibration_metrics(y_test, ref)["auroc"],
                    "delta_auprc": calibration_metrics(y_test, p)["auprc"] - calibration_metrics(y_test, ref)["auprc"],
                    "delta_brier_score": calibration_metrics(y_test, p)["brier_score"] - calibration_metrics(y_test, ref)["brier_score"],
                    "delta_absolute_calibration_error": calibration_metrics(y_test, p)["absolute_calibration_error"]
                    - calibration_metrics(y_test, ref)["absolute_calibration_error"],
                }
                seed = RANDOM_STATE + landmark * 1000 + zlib.crc32(method.encode()) % 1000
                delta.update(paired_delta_ci(y_test, ref, p, test_subjects, seed))
                delta_rows.append(delta)

    perf = pd.DataFrame(perf_rows)
    deltas = pd.DataFrame(delta_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    perf.to_csv(OUTPUT_DIR / "model_v13_recalibration_performance.csv", index=False)
    deltas.to_csv(OUTPUT_DIR / "model_v13_recalibration_paired_delta.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "model_v13_recalibrated_predictions_all_landmarks.csv", index=False)
    make_recalibration_figures(predictions, perf)
    return perf, deltas, predictions


def run_measurement_intensity_sensitivity(train_subjects: set[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    perf_rows = []
    delta_rows = []
    feature_rows = []
    for landmark in LANDMARKS:
        base_data = load_data(landmark)
        data = add_measurement_counts(base_data, landmark)
        train_mask = data["subject_id"].astype(int).isin(train_subjects)
        model_name = SELECTED_MODEL[landmark]
        variants = variant_predictor_sets(data, landmark)

        predictions: dict[str, np.ndarray] = {}
        y_test_reference: np.ndarray | None = None
        subject_reference: np.ndarray | None = None
        for variant, (predictors, add_indicators) in variants.items():
            if not predictors:
                continue
            y_test, p, subjects = train_predict_variant(
                data,
                train_mask,
                predictors,
                model_name,
                add_missing_indicators=add_indicators,
            )
            if y_test_reference is None:
                y_test_reference = y_test
                subject_reference = subjects
            elif not np.array_equal(y_test_reference, y_test):
                raise ValueError("Variant test outcomes are not aligned")
            predictions[variant] = p
            row = {
                "landmark_hours": landmark,
                "selected_model": model_name,
                "variant": variant,
                "add_imputation_missingness_indicators": add_indicators,
                "predictor_n": len(predictors),
                "per_feature_count_predictor_n": sum(
                    is_feature_count_column(c) and c not in aggregate_intensity_columns(data) for c in predictors
                ),
                "aggregate_intensity_predictor_n": len([c for c in predictors if c in aggregate_intensity_columns(data)]),
                "test_n": len(y_test),
                "test_event_n": int(y_test.sum()),
                "test_event_rate": float(y_test.mean()),
            }
            row.update(calibration_metrics(y_test, p))
            perf_rows.append(row)
            for predictor in predictors:
                feature_rows.append(
                    {
                        "landmark_hours": landmark,
                        "variant": variant,
                        "predictor": predictor,
                        "is_per_feature_count": is_feature_count_column(predictor)
                        and predictor not in aggregate_intensity_columns(data),
                        "is_aggregate_intensity": predictor in aggregate_intensity_columns(data),
                    }
                )

        assert y_test_reference is not None and subject_reference is not None
        reference = predictions["full_current"]
        for variant, p in predictions.items():
            if variant == "full_current":
                continue
            delta = {
                "landmark_hours": landmark,
                "selected_model": model_name,
                "variant": variant,
                "delta_auroc": calibration_metrics(y_test_reference, p)["auroc"]
                - calibration_metrics(y_test_reference, reference)["auroc"],
                "delta_auprc": calibration_metrics(y_test_reference, p)["auprc"]
                - calibration_metrics(y_test_reference, reference)["auprc"],
                "delta_brier_score": calibration_metrics(y_test_reference, p)["brier_score"]
                - calibration_metrics(y_test_reference, reference)["brier_score"],
                "delta_absolute_calibration_error": calibration_metrics(y_test_reference, p)["absolute_calibration_error"]
                - calibration_metrics(y_test_reference, reference)["absolute_calibration_error"],
            }
            seed = RANDOM_STATE + landmark * 1000 + zlib.crc32(variant.encode()) % 1000
            delta.update(paired_delta_ci(y_test_reference, reference, p, subject_reference, seed))
            delta_rows.append(delta)

        pred_table = data.loc[~train_mask, ["subject_id", "hadm_id", "stay_id", "landmark_hours"]].copy()
        pred_table["y_true"] = y_test_reference
        for variant, p in predictions.items():
            pred_table[f"prob_{variant}"] = p
        pred_table.to_csv(OUTPUT_DIR / f"model_v13_measurement_intensity_predictions_{landmark}h.csv", index=False)

    perf = pd.DataFrame(perf_rows)
    deltas = pd.DataFrame(delta_rows)
    features = pd.DataFrame(feature_rows)
    perf.to_csv(OUTPUT_DIR / "model_v13_measurement_intensity_sensitivity_performance.csv", index=False)
    deltas.to_csv(OUTPUT_DIR / "model_v13_measurement_intensity_paired_delta.csv", index=False)
    features.to_csv(OUTPUT_DIR / "audit_v13_measurement_intensity_predictor_sets.csv", index=False)
    make_measurement_figures(perf)
    return perf, deltas, features


def write_readme(recal_perf: pd.DataFrame, recal_delta: pd.DataFrame, mi_perf: pd.DataFrame, mi_delta: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# v13 Recalibration and measurement-intensity sensitivity")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This analysis does not change the primary v5-v12 results. It evaluates whether probability calibration can be improved using development-set recalibration and whether model performance depends materially on measurement-intensity information.")
    lines.append("")
    lines.append("## Calibration update")
    lines.append("")
    lines.append("Recalibrators were learned from grouped out-of-fold predictions in the development set and evaluated once in the held-out subject-level test set.")
    lines.append("")
    lines.append("| Landmark | Model | Method | AUROC | AUPRC | Brier | O/E | Calibration slope |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
    for row in recal_perf.sort_values(["landmark_hours", "recalibration_method"]).itertuples():
        lines.append(
            f"| {int(row.landmark_hours)} h | {row.selected_model} | {row.recalibration_method} | "
            f"{row.auroc:.3f} | {row.auprc:.3f} | {row.brier_score:.3f} | "
            f"{row.observed_expected_ratio:.2f} | {row.calibration_slope:.2f} |"
        )
    lines.append("")
    lines.append("Best Brier score by landmark:")
    for landmark, group in recal_perf.groupby("landmark_hours"):
        best = group.sort_values("brier_score").iloc[0]
        base = group[group["recalibration_method"].eq("none")].iloc[0]
        lines.append(
            f"- {int(landmark)} h: {best.recalibration_method} Brier {best.brier_score:.3f} "
            f"vs uncalibrated {base.brier_score:.3f}."
        )
    lines.append("")
    lines.append("## Measurement-intensity sensitivity")
    lines.append("")
    lines.append("Variants compare the current model against removal/addition of measurement-process variables. The main pipeline already includes imputation missingness indicators; one conservative variant disables them.")
    lines.append("")
    lines.append("| Landmark | Variant | Predictors | AUROC | AUPRC | Brier | O/E | Calibration slope |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in mi_perf.sort_values(["landmark_hours", "variant"]).itertuples():
        lines.append(
            f"| {int(row.landmark_hours)} h | {row.variant} | {int(row.predictor_n)} | "
            f"{row.auroc:.3f} | {row.auprc:.3f} | {row.brier_score:.3f} | "
            f"{row.observed_expected_ratio:.2f} | {row.calibration_slope:.2f} |"
        )
    lines.append("")
    lines.append("Key AUROC deltas versus current full model:")
    for row in mi_delta.sort_values(["landmark_hours", "variant"]).itertuples():
        lines.append(
            f"- {int(row.landmark_hours)} h / {row.variant}: ΔAUROC {row.delta_auroc:+.3f}, "
            f"ΔBrier {row.delta_brier_score:+.3f}."
        )
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("- `model_v13_recalibration_performance.csv`")
    lines.append("- `model_v13_recalibration_paired_delta.csv`")
    lines.append("- `model_v13_recalibrated_predictions_all_landmarks.csv`")
    lines.append("- `model_v13_measurement_intensity_sensitivity_performance.csv`")
    lines.append("- `model_v13_measurement_intensity_paired_delta.csv`")
    lines.append("- `audit_v13_measurement_intensity_predictor_sets.csv`")
    lines.append("- `figure_v13_recalibration_curve_0h.png`, `figure_v13_recalibration_curve_6h.png`, `figure_v13_recalibration_curve_24h.png`")
    lines.append("- `figure_v13_recalibration_brier_delta.png`")
    lines.append("- `figure_v13_measurement_intensity_auroc_delta.png`")
    lines.append("- `figure_v13_measurement_intensity_brier_delta.png`")
    (OUTPUT_DIR / "audit_v13_results_brief.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_0h = load_data(0)
    train_subjects, test_subjects, split_audit = choose_grouped_split(baseline_0h)
    pd.DataFrame([split_audit]).to_csv(OUTPUT_DIR / "audit_v13_subject_split.csv", index=False)

    recal_perf, recal_delta, _ = run_recalibration(train_subjects)
    mi_perf, mi_delta, _ = run_measurement_intensity_sensitivity(train_subjects)
    write_readme(recal_perf, recal_delta, mi_perf, mi_delta)
    print(f"v13 recalibration and measurement-intensity sensitivity complete: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
