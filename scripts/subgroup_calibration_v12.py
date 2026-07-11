"""Subgroup calibration audit for dynamic postoperative AKI models.

This script uses held-out test-set predictions from v5.1 and merges subgroup
variables from the v4.1 modeling-ready datasets. It does not refit models.

Outputs include subgroup calibration metrics, quantile-binned calibration
curves, and figures focused on the selected model at each landmark.
"""

from __future__ import annotations

from pathlib import Path
import zlib

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "modeling_v5_1"
MODELING_DIR = PROJECT_ROOT / "outputs" / "modeling_v4_1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v12_subgroup_calibration"

LANDMARKS = [0, 6, 24]
OUTCOME = "outcome_aki_after_landmark_to_7d"
RANDOM_STATE = 20260709
BOOTSTRAPS = 100
MIN_N = 80
MIN_EVENTS = 10
MIN_NONEVENTS = 10

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

SELECTED_MODEL = {
    0: "xgboost",
    6: "xgboost",
    24: "logistic_regression",
}

SUBGROUP_COLUMNS = [
    "first_careunit",
    "gender",
    "anchor_age",
    "race",
    "dm",
    "ckd",
    "cardiac_surgery",
    "non_cardiac_surgery",
    "vascular_surgery",
    "general_gi_hepatobiliary_surgery",
    "orthopedic_major_surgery",
    "neurosurgery",
    "thoracic_respiratory_surgery",
    "baseline_scr_source_at_landmark",
]

SURGERY_GROUPS = [
    "cardiac_surgery",
    "non_cardiac_surgery",
    "vascular_surgery",
    "general_gi_hepatobiliary_surgery",
    "orthopedic_major_surgery",
    "neurosurgery",
    "thoracic_respiratory_surgery",
]


def clamp_probability(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def logit(p: np.ndarray) -> np.ndarray:
    p = clamp_probability(p)
    return np.log(p / (1 - p))


def boolish(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(float).eq(1)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def safe_rate(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else np.nan


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Fit logit(y) = intercept + slope * logit(p) using scipy optimization."""
    y = np.asarray(y, dtype=float)
    x = logit(p)
    if len(np.unique(y)) < 2 or np.nanstd(x) == 0:
        return np.nan, np.nan

    try:
        from scipy.optimize import minimize

        def nll(beta: np.ndarray) -> float:
            eta = beta[0] + beta[1] * x
            # stable logistic negative log-likelihood
            return float(np.sum(np.logaddexp(0, eta) - y * eta))

        result = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
        if not result.success and not np.all(np.isfinite(result.x)):
            return np.nan, np.nan
        return float(result.x[0]), float(result.x[1])
    except Exception:
        return np.nan, np.nan


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = clamp_probability(p)
    observed = float(y.mean())
    mean_predicted = float(p.mean())
    intercept, slope = calibration_intercept_slope(y, p)
    metrics = {
        "observed_risk": observed,
        "mean_predicted_risk": mean_predicted,
        "absolute_calibration_error": abs(observed - mean_predicted),
        "observed_expected_ratio": observed / mean_predicted if mean_predicted > 0 else np.nan,
        "calibration_in_large": logit(np.array([observed]))[0] - logit(np.array([mean_predicted]))[0]
        if 0 < observed < 1 and 0 < mean_predicted < 1
        else np.nan,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "brier_score": brier_score_loss(y, p),
    }
    if len(np.unique(y)) > 1:
        metrics["auroc"] = roc_auc_score(y, p)
        metrics["auprc"] = average_precision_score(y, p)
    else:
        metrics["auroc"] = np.nan
        metrics["auprc"] = np.nan
    return metrics


def subject_bootstrap_indices(subjects: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(subjects)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    lookup = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    return np.concatenate([lookup[subject] for subject in sampled])


def bootstrap_metric_ci(y: np.ndarray, p: np.ndarray, subjects: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = {
        "observed_expected_ratio": [],
        "calibration_in_large": [],
        "calibration_slope": [],
        "brier_score": [],
    }
    attempts = 0
    while len(values["brier_score"]) < BOOTSTRAPS and attempts < BOOTSTRAPS * 4:
        attempts += 1
        idx = subject_bootstrap_indices(subjects, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        m = calibration_metrics(y[idx], p[idx])
        for key in values:
            if np.isfinite(m[key]):
                values[key].append(m[key])
    out: dict[str, float] = {"bootstrap_successful_n": len(values["brier_score"])}
    for key, vals in values.items():
        if vals:
            out[f"{key}_ci_lower"] = float(np.quantile(vals, 0.025))
            out[f"{key}_ci_upper"] = float(np.quantile(vals, 0.975))
        else:
            out[f"{key}_ci_lower"] = np.nan
            out[f"{key}_ci_upper"] = np.nan
    return out


def empty_bootstrap_ci() -> dict[str, float]:
    out: dict[str, float] = {"bootstrap_successful_n": 0}
    for key in ["observed_expected_ratio", "calibration_in_large", "calibration_slope", "brier_score"]:
        out[f"{key}_ci_lower"] = np.nan
        out[f"{key}_ci_upper"] = np.nan
    return out


def load_landmark_data(landmark: int) -> pd.DataFrame:
    pred_path = PREDICTION_DIR / f"model_v5_1_{landmark}h_test_predictions.csv"
    model_path = MODELING_DIR / f"modeling_v4_1_{landmark}h.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    pred = pd.read_csv(pred_path, low_memory=False)
    model = pd.read_csv(model_path, usecols=lambda c: c in {"subject_id", "hadm_id", "stay_id", *SUBGROUP_COLUMNS}, low_memory=False)
    merged = pred.merge(model, on=["subject_id", "hadm_id", "stay_id"], how="left", validate="one_to_one")
    if merged[SUBGROUP_COLUMNS].isna().all(axis=None):
        raise ValueError(f"No subgroup variables merged for {landmark}h")
    return merged


def add_derived_subgroups(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    age = pd.to_numeric(out["anchor_age"], errors="coerce")
    out["age_group"] = pd.cut(
        age,
        bins=[-np.inf, 64, 74, np.inf],
        labels=["<65", "65-74", ">=75"],
    ).astype("string").fillna("<missing>")
    out["ckd_status"] = np.where(boolish(out["ckd"]), "CKD", "No CKD")
    out["diabetes_status"] = np.where(boolish(out["dm"]), "Diabetes", "No diabetes")
    out["cardiac_status"] = np.where(boolish(out["cardiac_surgery"]), "Cardiac surgery", "Non-cardiac surgery")
    return out


def subgroup_definitions(data: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    defs: list[tuple[str, str, pd.Series]] = []

    for col in ["gender", "age_group", "ckd_status", "diabetes_status", "cardiac_status"]:
        for value, n in data[col].fillna("<missing>").value_counts().items():
            if n >= MIN_N:
                defs.append((col, str(value), data[col].fillna("<missing>").eq(value)))

    for col in SURGERY_GROUPS:
        mask = boolish(data[col])
        defs.append((col, "yes", mask))

    for col in ["first_careunit", "baseline_scr_source_at_landmark"]:
        counts = data[col].fillna("<missing>").astype(str).value_counts()
        for value, n in counts.items():
            if n >= MIN_N:
                defs.append((col, str(value), data[col].fillna("<missing>").astype(str).eq(str(value))))

    # Race can be granular in MIMIC; keep only larger categories to avoid unstable calibration.
    for value, n in data["race"].fillna("<missing>").astype(str).value_counts().items():
        if n >= 150:
            defs.append(("race", str(value), data["race"].fillna("<missing>").astype(str).eq(str(value))))

    return defs


def subgroup_calibration_rows(landmark: int, data: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prob_cols = [c for c in data.columns if c.startswith("prob_")]
    for dimension, level, mask in subgroup_definitions(data):
        selected = mask.to_numpy(dtype=bool)
        y = data.loc[selected, "y_true"].to_numpy(dtype=int)
        subjects = data.loc[selected, "subject_id"].to_numpy(dtype=int)
        if len(y) < MIN_N or y.sum() < MIN_EVENTS or (len(y) - y.sum()) < MIN_NONEVENTS:
            continue
        for prob_col in prob_cols:
            model_key = prob_col.replace("prob_", "")
            p = data.loc[selected, prob_col].to_numpy(dtype=float)
            point = calibration_metrics(y, p)
            seed = RANDOM_STATE + landmark * 10000 + zlib.crc32(f"{dimension}|{level}|{model_key}".encode()) % 10000
            if model_key == SELECTED_MODEL[landmark]:
                ci = bootstrap_metric_ci(y, p, subjects, seed)
            else:
                ci = empty_bootstrap_ci()
            rows.append(
                {
                    "landmark_hours": landmark,
                    "model_key": model_key,
                    "model": MODEL_LABELS.get(model_key, model_key),
                    "selected_model_for_landmark": model_key == SELECTED_MODEL[landmark],
                    "subgroup_dimension": dimension,
                    "subgroup_level": level,
                    "n": int(len(y)),
                    "subject_n": int(len(np.unique(subjects))),
                    "event_n": int(y.sum()),
                    "event_rate": float(y.mean()),
                    **point,
                    **ci,
                }
            )
    return rows


def calibration_bins(landmark: int, data: pd.DataFrame, selected_only: bool = False) -> pd.DataFrame:
    rows = []
    model_keys = [SELECTED_MODEL[landmark]] if selected_only else [
        c.replace("prob_", "") for c in data.columns if c.startswith("prob_")
    ]
    for dimension, level, mask in subgroup_definitions(data):
        selected = mask.to_numpy(dtype=bool)
        subgroup = data.loc[selected].copy()
        if len(subgroup) < max(150, MIN_N) or subgroup["y_true"].sum() < MIN_EVENTS:
            continue
        for model_key in model_keys:
            prob_col = f"prob_{model_key}"
            temp = subgroup[["y_true", prob_col]].dropna().copy()
            if temp[prob_col].nunique() < 4:
                continue
            try:
                temp["bin"] = pd.qcut(temp[prob_col], q=min(5, temp[prob_col].nunique()), duplicates="drop")
            except ValueError:
                continue
            grouped = temp.groupby("bin", observed=True)
            for bin_id, (_, g) in enumerate(grouped, start=1):
                rows.append(
                    {
                        "landmark_hours": landmark,
                        "model_key": model_key,
                        "model": MODEL_LABELS.get(model_key, model_key),
                        "selected_model_for_landmark": model_key == SELECTED_MODEL[landmark],
                        "subgroup_dimension": dimension,
                        "subgroup_level": level,
                        "bin": bin_id,
                        "n": len(g),
                        "mean_predicted_risk": float(g[prob_col].mean()),
                        "observed_risk": float(g["y_true"].mean()),
                        "predicted_min": float(g[prob_col].min()),
                        "predicted_max": float(g[prob_col].max()),
                    }
                )
    return pd.DataFrame(rows)


def save_figures(metrics: pd.DataFrame, bins: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    selected = metrics.loc[metrics["selected_model_for_landmark"]].copy()
    selected["subgroup_label"] = selected["subgroup_dimension"] + ": " + selected["subgroup_level"]

    # Figure 1: O/E ratio forest plot for selected models and common surgical groups.
    surgery = selected.loc[selected["subgroup_dimension"].isin(SURGERY_GROUPS)].copy()
    surgery = surgery.sort_values(["landmark_hours", "subgroup_dimension"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.0), sharex=True)
    for ax, landmark in zip(axes, LANDMARKS):
        panel = surgery.loc[surgery["landmark_hours"].eq(landmark)].copy()
        panel = panel.sort_values("observed_expected_ratio")
        y_pos = np.arange(len(panel))
        ax.errorbar(
            panel["observed_expected_ratio"],
            y_pos,
            xerr=[
                panel["observed_expected_ratio"] - panel["observed_expected_ratio_ci_lower"],
                panel["observed_expected_ratio_ci_upper"] - panel["observed_expected_ratio"],
            ],
            fmt="o",
            color="#4c78a8",
            ecolor="#9ecae9",
            capsize=2,
        )
        ax.axvline(1.0, color="black", linestyle="--", linewidth=0.9)
        ax.set_title(f"{landmark} h")
        ax.set_yticks(y_pos, labels=panel["subgroup_dimension"].str.replace("_", " "))
        ax.set_xlabel("Observed / expected risk")
    axes[0].set_ylabel("Surgical subgroup")
    fig.suptitle("Subgroup calibration: observed-to-expected risk ratio for selected models", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "figure_v12_selected_model_subgroup_oe_ratio.png", dpi=300)
    plt.close(fig)

    # Figure 2: calibration slope heatmap for selected model.
    heat = selected.pivot_table(
        index="subgroup_label",
        columns="landmark_hours",
        values="calibration_slope",
        aggfunc="first",
    )
    keep = heat.notna().sum(axis=1).sort_values(ascending=False).head(28).index
    heat = heat.loc[keep]
    fig, ax = plt.subplots(figsize=(7.8, max(5.5, len(heat) * 0.25)))
    im = ax.imshow(heat.values, cmap="coolwarm", aspect="auto", vmin=0.4, vmax=1.6)
    ax.set_xticks(np.arange(len(heat.columns)), labels=[f"{int(c)} h" for c in heat.columns])
    ax.set_yticks(np.arange(len(heat.index)), labels=heat.index)
    ax.set_title("Calibration slope by subgroup for selected models")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            ax.text(j, i, "" if pd.isna(val) else f"{val:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Calibration slope")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_v12_selected_model_subgroup_calibration_slope_heatmap.png", dpi=300)
    plt.close(fig)

    # Figure 3: binned calibration curves for cardiac/non-cardiac selected models.
    curve = bins.loc[
        bins["selected_model_for_landmark"]
        & bins["subgroup_dimension"].eq("cardiac_status")
    ].copy()
    if not curve.empty:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharex=True, sharey=True)
        colors = {"Cardiac surgery": "#e45756", "Non-cardiac surgery": "#4c78a8"}
        for ax, landmark in zip(axes, LANDMARKS):
            panel = curve.loc[curve["landmark_hours"].eq(landmark)]
            for level, g in panel.groupby("subgroup_level"):
                ax.plot(
                    g["mean_predicted_risk"],
                    g["observed_risk"],
                    marker="o",
                    linewidth=1.5,
                    label=level,
                    color=colors.get(level, None),
                )
            ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.9)
            ax.set_title(f"{landmark} h")
            ax.set_xlabel("Mean predicted risk")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("Observed risk")
        axes[-1].legend(frameon=False, loc="lower right")
        fig.suptitle("Binned subgroup calibration curves: cardiac vs non-cardiac surgery", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(OUTPUT_DIR / "figure_v12_cardiac_noncardiac_calibration_curves.png", dpi=300)
        plt.close(fig)


def write_readme(metrics: pd.DataFrame) -> None:
    selected = metrics.loc[metrics["selected_model_for_landmark"]].copy()
    overall_lines = []
    for landmark in LANDMARKS:
        panel = selected.loc[selected["landmark_hours"].eq(landmark)]
        worst_oe = panel.assign(abs_log_oe=lambda d: np.abs(np.log(d["observed_expected_ratio"]))).sort_values(
            "abs_log_oe", ascending=False
        ).head(5)
        overall_lines.append(f"### {landmark} h selected model ({MODEL_LABELS[SELECTED_MODEL[landmark]]})")
        for row in worst_oe.itertuples():
            overall_lines.append(
                f"- {row.subgroup_dimension} = {row.subgroup_level}: n={row.n:,}, event rate={row.event_rate:.3f}, "
                f"mean predicted={row.mean_predicted_risk:.3f}, O/E={row.observed_expected_ratio:.2f}, "
                f"slope={row.calibration_slope:.2f}, Brier={row.brier_score:.3f}."
            )

    content = f"""# v12 Subgroup calibration audit

## Scope

This audit extends subgroup evaluation beyond AUROC by assessing probability calibration in held-out v5.1 test-set predictions. No models were refit.

## Metrics

For each eligible subgroup, the audit reports:

- observed event rate;
- mean predicted risk;
- observed-to-expected risk ratio;
- calibration-in-the-large;
- calibration intercept and calibration slope;
- Brier score;
- AUROC and AUPRC for context.

Subgroups with fewer than {MIN_N} patients, fewer than {MIN_EVENTS} events, or fewer than {MIN_NONEVENTS} non-events were excluded from calibration estimates.

## Selected-model calibration signals

{chr(10).join(overall_lines)}

## Interpretation

Subgroup calibration should be interpreted separately from subgroup discrimination. A subgroup can have acceptable AUROC while systematically overpredicting or underpredicting absolute risk. These results are most appropriate for supplementary tables/figures and for a Discussion limitation on subgroup-specific reliability of predicted probabilities.

## Output files

- `audit_v12_subgroup_calibration_metrics.csv`
- `audit_v12_subgroup_calibration_bins.csv`
- `audit_v12_selected_model_subgroup_calibration.csv`
- `figure_v12_selected_model_subgroup_oe_ratio.png`
- `figure_v12_selected_model_subgroup_calibration_slope_heatmap.png`
- `figure_v12_cardiac_noncardiac_calibration_curves.png`
"""
    (OUTPUT_DIR / "audit_v12_results_brief.md").write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    bin_tables: list[pd.DataFrame] = []

    for landmark in LANDMARKS:
        data = add_derived_subgroups(load_landmark_data(landmark))
        metric_rows.extend(subgroup_calibration_rows(landmark, data))
        bin_tables.append(calibration_bins(landmark, data, selected_only=False))

    metrics = pd.DataFrame(metric_rows)
    bins = pd.concat(bin_tables, ignore_index=True)
    metrics.to_csv(OUTPUT_DIR / "audit_v12_subgroup_calibration_metrics.csv", index=False)
    bins.to_csv(OUTPUT_DIR / "audit_v12_subgroup_calibration_bins.csv", index=False)
    metrics.loc[metrics["selected_model_for_landmark"]].to_csv(
        OUTPUT_DIR / "audit_v12_selected_model_subgroup_calibration.csv", index=False
    )

    save_figures(metrics, bins)
    write_readme(metrics)
    print(f"v12 subgroup calibration audit complete: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
