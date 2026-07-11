"""v15 deeper secondary analyses for dynamic postoperative AKI prediction.

This script deliberately keeps the primary prediction analysis unchanged and
adds two clearly secondary analyses:

1. Target-trial-inspired, overlap-weighted risk contrasts for early,
   potentially actionable physiological states at the 6 h and 24 h landmarks.
   These are observational estimands and are explicitly not causal treatment
   effects: no treatment assignment, no unmeasured-confounding guarantee, and
   no time-varying treatment model are available in the current data.
2. Threshold-policy summaries for the locked parsimonious prediction models,
   using their held-out test predictions. These quantify alert burden and
   captured events at practical absolute-risk thresholds.

Inputs are existing v4.1 modelling-ready datasets and v10 held-out
predictions. No MIMIC raw tables are read.
"""

from __future__ import annotations

from pathlib import Path
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from develop_models_v5 import OUTCOME, RANDOM_STATE, load_data  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v15_target_trial_clinical_utility"
V10_DIR = PROJECT_ROOT / "outputs" / "modeling_v10_simplified_model"

SELECTED_PROBABILITY = {
    0: "prob_simplified_xgboost",
    6: "prob_simplified_xgboost",
    24: "prob_simplified_logistic_regression",
}

# Baseline covariates selected before any landmark-specific exposure window.
# No landmark-window laboratory/vital-sign measure is adjusted for, because it
# could be downstream of or concurrent with the exposed physiological state.
BASELINE_COVARIATES = [
    "anchor_age", "gender", "race", "admission_type", "first_careunit",
    "cardiac_surgery", "vascular_surgery", "general_gi_hepatobiliary_surgery",
    "orthopedic_major_surgery", "neurosurgery", "thoracic_respiratory_surgery",
    "chf", "hypertension", "dm", "ckd", "charlson_score",
    "baseline_scr_at_landmark", "lab_pre24h_hemoglobin_last",
    "lab_pre24h_lactate_last", "lab_pre24h_bun_last",
]

EXPOSURES = {
    6: [
        ("early_map_lt_65", "0–6 h minimum MAP <65 mmHg", "vital_0_6h_map_min", "lt", 65.0, None),
        ("early_lactate_ge_2", "0–6 h maximum lactate ≥2 mmol/L", "lab_0_6h_lactate_max", "ge", 2.0, None),
        ("early_potassium_abnormal", "0–6 h potassium outside 3.5–5.0 mmol/L", "lab_0_6h_potassium_max", "outside", 5.0, "lab_0_6h_potassium_min"),
    ],
    24: [
        ("early_map_lt_65", "0–24 h minimum MAP <65 mmHg", "vital_0_24h_map_min", "lt", 65.0, None),
        ("early_lactate_ge_2", "0–24 h maximum lactate ≥2 mmol/L", "lab_0_24h_lactate_max", "ge", 2.0, None),
        ("early_potassium_abnormal", "0–24 h potassium outside 3.5–5.0 mmol/L", "lab_0_24h_potassium_max", "outside", 5.0, "lab_0_24h_potassium_min"),
    ],
}


def bool_series(x: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(x):
        return x.fillna(False).astype(int)
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce").fillna(0).astype(int)
    return x.astype("string").str.lower().isin(["true", "1", "yes"]).astype(int)


def make_exposure(data: pd.DataFrame, source: str, kind: str, value: float, second_source: str | None) -> pd.Series:
    high = pd.to_numeric(data[source], errors="coerce")
    if kind == "lt":
        return high.lt(value).astype(float).where(high.notna())
    if kind == "ge":
        return high.ge(value).astype(float).where(high.notna())
    if kind == "outside":
        low = pd.to_numeric(data[second_source], errors="coerce")
        observed = high.notna() | low.notna()
        return (high.gt(value) | low.lt(3.5)).astype(float).where(observed)
    raise ValueError(kind)


def split_covariates(work: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    categorical, numeric = [], []
    for col in cols:
        if pd.api.types.is_bool_dtype(work[col]) or pd.api.types.is_numeric_dtype(work[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def make_propensity_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers, remainder="drop")),
        ("model", LogisticRegression(C=0.5, max_iter=2500, solver="lbfgs", random_state=RANDOM_STATE)),
    ])


def oof_propensity(work: pd.DataFrame, covariates: list[str], exposure: np.ndarray) -> np.ndarray:
    numeric, categorical = split_covariates(work, covariates)
    groups = pd.to_numeric(work["subject_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
    counts = pd.Series(exposure).value_counts()
    group_n = pd.Series(groups).nunique()
    folds = min(5, int(counts.min()), group_n)
    if folds < 2:
        model = make_propensity_pipeline(numeric, categorical)
        model.fit(work[covariates], exposure)
        return model.predict_proba(work[covariates])[:, 1]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    estimated = np.full(len(work), np.nan)
    for train_idx, test_idx in splitter.split(work[covariates], exposure, groups):
        model = make_propensity_pipeline(numeric, categorical)
        model.fit(work.iloc[train_idx][covariates], exposure[train_idx])
        estimated[test_idx] = model.predict_proba(work.iloc[test_idx][covariates])[:, 1]
    if np.isnan(estimated).any():
        raise RuntimeError("Cross-fitted propensity scores were incomplete")
    return estimated


def overlap_estimate(y: np.ndarray, a: np.ndarray, e: np.ndarray) -> dict[str, float]:
    # The overlap population downweights patients with near-deterministic
    # exposure assignments and targets patients for whom both states were
    # empirically plausible from measured baseline risk factors.
    w1 = 1.0 - e[a == 1]
    w0 = e[a == 0]
    risk1 = float(np.sum(w1 * y[a == 1]) / np.sum(w1))
    risk0 = float(np.sum(w0 * y[a == 0]) / np.sum(w0))
    return {"risk_exposed": risk1, "risk_unexposed": risk0, "risk_difference": risk1 - risk0,
            "risk_ratio": risk1 / risk0 if risk0 > 0 else np.nan}


def standardized_mean_difference(a: np.ndarray, x: np.ndarray, e: np.ndarray | None = None) -> float:
    if e is None:
        w = np.ones_like(a, dtype=float)
    else:
        w = np.where(a == 1, 1.0 - e, e)
    mask1, mask0 = a == 1, a == 0
    def wmean(v, wt): return np.sum(v * wt) / np.sum(wt)
    def wvar(v, wt, mean): return np.sum(wt * (v - mean) ** 2) / np.sum(wt)
    m1, m0 = wmean(x[mask1], w[mask1]), wmean(x[mask0], w[mask0])
    v1, v0 = wvar(x[mask1], w[mask1], m1), wvar(x[mask0], w[mask0], m0)
    return float((m1 - m0) / np.sqrt(max((v1 + v0) / 2, 1e-10)))


def cluster_bootstrap_ci(work: pd.DataFrame, y: np.ndarray, a: np.ndarray, e: np.ndarray, draws: int = 500) -> tuple[float, float]:
    # Cluster resampling preserves repeated ICU admissions from individual
    # subjects. Propensity scores are held fixed, so intervals quantify sampling
    # variability conditional on the cross-fitted nuisance model.
    rng = np.random.default_rng(RANDOM_STATE + len(work))
    group_rows = work.groupby("subject_id", sort=False).indices
    groups = list(group_rows)
    estimates = []
    for _ in range(draws):
        selected = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([group_rows[g] for g in selected])
        if len(np.unique(a[idx])) < 2:
            continue
        estimates.append(overlap_estimate(y[idx], a[idx], e[idx])["risk_difference"])
    if not estimates:
        return np.nan, np.nan
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def propensity_balance(work: pd.DataFrame, covariates: list[str], a: np.ndarray, e: np.ndarray) -> tuple[float, float]:
    # Report balance across numeric covariates only; categorical balance is
    # handled inside the propensity model but is not condensed into a misleading
    # single categorical SMD here.
    numeric = [c for c in covariates if pd.api.types.is_numeric_dtype(work[c]) or pd.api.types.is_bool_dtype(work[c])]
    before, after = [], []
    for col in numeric:
        x = pd.to_numeric(work[col], errors="coerce")
        if x.notna().sum() < 100:
            continue
        x = x.fillna(x.median()).to_numpy(dtype=float)
        before.append(abs(standardized_mean_difference(a, x)))
        after.append(abs(standardized_mean_difference(a, x, e)))
    return (float(max(before)) if before else np.nan, float(max(after)) if after else np.nan)


def target_trial_inspired_analysis() -> pd.DataFrame:
    rows = []
    for landmark, specs in EXPOSURES.items():
        data = load_data(landmark).copy()
        covariates = [x for x in BASELINE_COVARIATES if x in data.columns]
        for name, label, source, kind, value, second_source in specs:
            exposure = make_exposure(data, source, kind, value, second_source)
            work = data[["subject_id", OUTCOME, *covariates]].copy()
            work["exposure"] = exposure
            work = work.dropna(subset=[OUTCOME, "exposure"]).reset_index(drop=True)
            work["exposure"] = work["exposure"].astype(int)
            y = bool_series(work[OUTCOME]).to_numpy()
            a = work["exposure"].to_numpy()
            if len(work) < 500 or len(np.unique(a)) < 2 or min(a.sum(), len(a) - a.sum()) < 50:
                continue
            e = np.clip(oof_propensity(work, covariates, a), 0.01, 0.99)
            # Restrict to prespecified overlap support; this is transparent and
            # avoids extrapolation for nearly deterministic physiological states.
            supported = (e >= 0.05) & (e <= 0.95)
            ws, ys, aas, es = work.loc[supported].copy(), y[supported], a[supported], e[supported]
            estimate = overlap_estimate(ys, aas, es)
            ci_low, ci_high = cluster_bootstrap_ci(ws, ys, aas, es)
            smd_before, smd_after = propensity_balance(ws, covariates, aas, es)
            rows.append({
                "landmark_hours": landmark,
                "physiological_state": name,
                "state_label": label,
                "outcome": "incident AKI after landmark through day 7",
                "source_variable": source,
                "eligible_measured_n": int(len(work)),
                "overlap_population_n": int(len(ws)),
                "overlap_population_percent": float(100 * len(ws) / len(work)),
                "event_n": int(ys.sum()),
                "event_rate": float(ys.mean()),
                "state_prevalence_percent": float(100 * aas.mean()),
                "risk_if_state_present": estimate["risk_exposed"],
                "risk_if_state_absent": estimate["risk_unexposed"],
                "overlap_weighted_risk_difference": estimate["risk_difference"],
                "risk_difference_ci_lower": ci_low,
                "risk_difference_ci_upper": ci_high,
                "overlap_weighted_risk_ratio": estimate["risk_ratio"],
                "max_abs_numeric_smd_before": smd_before,
                "max_abs_numeric_smd_after": smd_after,
                "propensity_method": "5-fold subject-grouped cross-fitted logistic propensity model; overlap weighting; support restricted to propensity 0.05–0.95",
                "interpretation": "Secondary observational, target-trial-inspired risk contrast. It is not a causal effect of treating or changing this state; residual/time-varying confounding and measurement-selection bias remain possible.",
            })
    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_DIR / "analysis_v15_target_trial_inspired_risk_contrasts.csv", index=False)
    return output


def net_benefit(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    alert = p >= threshold
    tp = np.sum((alert == 1) & (y == 1))
    fp = np.sum((alert == 1) & (y == 0))
    return float(tp / len(y) - fp / len(y) * threshold / (1 - threshold))


def clinical_utility_policy() -> pd.DataFrame:
    rows = []
    fixed_thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    for landmark, probability_col in SELECTED_PROBABILITY.items():
        pred = pd.read_csv(V10_DIR / f"model_v10_{landmark}h_full_vs_simplified_predictions.csv")
        y = pred["y_true"].astype(int).to_numpy()
        p = pd.to_numeric(pred[probability_col], errors="coerce").to_numpy()
        thresholds = fixed_thresholds + [float(np.quantile(p, q)) for q in (0.70, 0.80, 0.90)]
        for threshold in sorted(set(round(x, 5) for x in thresholds if 0 < x < 1)):
            alert = p >= threshold
            tp = int(np.sum(alert & (y == 1)))
            fp = int(np.sum(alert & (y == 0)))
            fn = int(np.sum(~alert & (y == 1)))
            tn = int(np.sum(~alert & (y == 0)))
            rows.append({
                "landmark_hours": landmark,
                "selected_model": "XGBoost" if landmark in (0, 6) else "Logistic Regression",
                "threshold": threshold,
                "threshold_type": "fixed" if threshold in fixed_thresholds else "empirical alert-volume",
                "test_n": int(len(y)),
                "event_n": int(y.sum()),
                "alert_n": int(alert.sum()),
                "alert_rate": float(alert.mean()),
                "true_positive_n": tp,
                "false_positive_n": fp,
                "false_negative_n": fn,
                "true_negative_n": tn,
                "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
                "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
                "positive_predictive_value": float(tp / (tp + fp)) if tp + fp else np.nan,
                "false_alerts_per_100_patients": float(100 * fp / len(y)),
                "alerts_per_100_patients": float(100 * alert.mean()),
                "events_captured_per_100_patients": float(100 * tp / len(y)),
                "net_benefit": net_benefit(y, p, threshold),
            })
    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_DIR / "analysis_v15_clinical_threshold_policy.csv", index=False)
    return output


def make_figures(contrast: pd.DataFrame, policy: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    if not contrast.empty:
        plot = contrast.copy().sort_values(["landmark_hours", "overlap_weighted_risk_difference"])
        labels = [f"{int(r.landmark_hours)} h: {r.state_label}" for r in plot.itertuples()]
        y = np.arange(len(plot))
        fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(plot) * 0.55)), dpi=180)
        ax.errorbar(
            plot["overlap_weighted_risk_difference"] * 100, y,
            xerr=[(plot["overlap_weighted_risk_difference"] - plot["risk_difference_ci_lower"]) * 100,
                  (plot["risk_difference_ci_upper"] - plot["overlap_weighted_risk_difference"]) * 100],
            fmt="o", color="#d95f02", ecolor="#f2b480", capsize=3,
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
        ax.set_yticks(y, labels=labels)
        ax.set_xlabel("Overlap-weighted absolute risk difference (percentage points)")
        ax.set_title("Secondary target-trial-inspired risk contrasts")
        fig.text(
            0.5, 0.925,
            "Observed physiological states at 6 h or 24 h; 95% subject-cluster bootstrap CI; observational analysis",
            ha="center", va="center", fontsize=9, color="#5f6675",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        fig.savefig(OUTPUT_DIR / "figure_v15_target_trial_inspired_risk_contrasts.png", dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), dpi=180, sharey=True)
    for ax, (landmark, group) in zip(axes, policy.groupby("landmark_hours", sort=True)):
        # The visual shows only fixed risk thresholds; empirical alert-volume
        # threshold rows remain in the accompanying table for operational use.
        group = group.loc[group["threshold_type"].eq("fixed")].sort_values("threshold")
        ax.plot(group["alerts_per_100_patients"], group["events_captured_per_100_patients"], marker="o", color="#4c78a8")
        for r in group.itertuples():
            ax.annotate(f"{r.threshold:.1f}", (r.alerts_per_100_patients, r.events_captured_per_100_patients), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_title(f"{int(landmark)} h")
        ax.set_xlabel("Alerts per 100 patients")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AKI events captured per 100 patients")
    fig.suptitle("Operational trade-off for selected parsimonious models", y=1.04)
    fig.text(0.5, 0.975, "Held-out test predictions; labels indicate absolute-risk alert thresholds", ha="center", va="center", fontsize=9, color="#5f6675")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "figure_v15_clinical_threshold_policy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_readme(contrast: pd.DataFrame, policy: pd.DataFrame) -> None:
    lines = ["# v15 deeper secondary analyses", ""]
    lines.extend([
        "## Scope", "",
        "These analyses do not modify the prespecified SCr-based primary outcome, development split, selected parsimonious models, or primary performance results.",
        "", "## Target-trial-inspired physiological-state analysis", "",
        "At 6 h and 24 h, each state was evaluated only among patients with that measurement available. Baseline covariates were modeled using five-fold subject-grouped cross-fitted logistic propensity scores. Results use overlap weighting and restrict to propensity-score support 0.05–0.95. Confidence intervals use 500 subject-cluster bootstrap resamples conditional on the fitted cross-fitted propensity scores.",
        "", "Crucially, these are not causal intervention effects. MAP, lactate, and potassium are physiological states/markers, not randomized treatments; unmeasured and time-varying confounding, indication bias, and measurement-selection bias remain.",
        "", "## Clinical threshold policy", "",
        "Threshold-policy results use held-out predictions of the locked parsimonious models (0 h XGBoost, 6 h XGBoost, 24 h logistic regression). They are descriptive operational trade-offs, not a recommendation to deploy a specific alert threshold without prospective workflow testing.",
        "", "## Headline results", "",
    ])
    if not contrast.empty:
        for r in contrast.itertuples():
            lines.append(f"- {r.state_label} at {int(r.landmark_hours)} h: overlap-weighted AKI risk difference {r.overlap_weighted_risk_difference*100:+.1f} percentage points (95% CI {r.risk_difference_ci_lower*100:+.1f} to {r.risk_difference_ci_upper*100:+.1f}); overlap sample n={r.overlap_population_n:,}.")
    lines.append("")
    lines.append("For the complete threshold-by-landmark counts, predictive values, false alerts, and net benefit, see `analysis_v15_clinical_threshold_policy.csv`.")
    (OUTPUT_DIR / "audit_v15_results_brief.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contrast = target_trial_inspired_analysis()
    policy = clinical_utility_policy()
    make_figures(contrast, policy)
    write_readme(contrast, policy)
    print(f"Wrote v15 outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
