"""Audit recovery-outcome observability and selection in the v27 AKI-onset cohort.

This secondary analysis does not redefine the locked primary AKI outcome.  It
quantifies who can be evaluated for persistence/recovery, uses cross-fitted
inverse-probability weights for outcome-model performance, and describes death
and live discharge as competing events rather than coding missing renal status
as recovery or non-recovery.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from develop_models_v5 import RANDOM_STATE  # noqa: E402
from final_sensitivity_and_actionable_analysis_v14 import model_pipeline  # noqa: E402
from recalibration_and_measurement_intensity_v13 import identify_types_for_columns  # noqa: E402


V26 = ROOT / "outputs" / "modeling_v26_aki_severity_trajectories"
V27 = ROOT / "outputs" / "modeling_v27_severity_recovery"
OUT = ROOT / "outputs" / "modeling_v28_recovery_observability"
ONSET_FILE = V27 / "dataset_v27_aki_onset_trajectory.csv"
COHORT_FILE = V26 / "cohort_v26_strict_aki_severity_recovery.csv"
PRED_FILE = V27 / "model_v27_test_predictions.csv"
SPLIT_FILE = V27 / "audit_v27_subject_split_assignment.csv"
MISSING_FILE = V27 / "audit_v27_predictor_missingness.csv"

TASKS = {
    "persistent_aki": {
        "evaluable": "persistence_evaluable",
        "outcome": "outcome_persistent_aki_beyond_48h",
        "selected_model": "XGBoost",
        "label": "Persistent AKI beyond 48 h",
    },
    "nonrecovery": {
        "evaluable": "end_recovery_evaluable",
        "outcome": "outcome_not_recovered_at_end",
        "selected_model": "Logistic Regression",
        "label": "Not recovered at observation end",
    },
}


def bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).astype(bool)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in {
        "png": {"dpi": 300},
        "pdf": {},
        "svg": {},
        "tiff": {"dpi": 600},
    }.items():
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    onset = pd.read_csv(ONSET_FILE, low_memory=False)
    cohort = pd.read_csv(COHORT_FILE, low_memory=False)
    predictions = pd.read_csv(PRED_FILE, low_memory=False)
    split = pd.read_csv(SPLIT_FILE, low_memory=False)
    missing = pd.read_csv(MISSING_FILE)
    predictors = missing.loc[missing["dataset"].eq("aki_onset"), "predictor"].tolist()
    if not predictors or (set(predictors) - set(onset.columns)):
        raise ValueError("v27 onset predictor dictionary is incomplete")
    if onset["stay_id"].duplicated().any():
        raise ValueError("AKI-onset dataset contains duplicate stays")
    dates = ["intime", "dischtime", "aki_onset_time_final", "end_observation_time", "dod"]
    for column in dates:
        if column in cohort:
            cohort[column] = pd.to_datetime(cohort[column], errors="coerce")
    return onset, cohort, predictions, split, predictors


def simple_metrics(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if weights is None:
        weights = np.ones(len(y), dtype=float)
    weights = np.asarray(weights, dtype=float)
    result = {
        "n": len(y),
        "event_n": int(y.sum()),
        "event_percent_unweighted": 100 * y.mean(),
        "event_percent_weighted": 100 * np.average(y, weights=weights),
        "auroc": roc_auc_score(y, p, sample_weight=weights),
        "auprc": average_precision_score(y, p, sample_weight=weights),
        "brier_score": np.average((y - p) ** 2, weights=weights),
        "mean_predicted_risk_weighted": np.average(p, weights=weights),
    }
    p_for_calibration = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p_for_calibration / (1 - p_for_calibration)).reshape(-1, 1)
    try:
        recal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        recal.fit(logit, y, sample_weight=weights)
        result["calibration_intercept"] = float(recal.intercept_[0])
        result["calibration_slope"] = float(recal.coef_[0, 0])
    except Exception:
        result["calibration_intercept"] = np.nan
        result["calibration_slope"] = np.nan
    return result


def crossfit_observability(
    data: pd.DataFrame, predictors: list[str], evaluable_column: str, seed: int
) -> tuple[np.ndarray, pd.DataFrame]:
    y = bool_mask(data[evaluable_column]).astype(int).to_numpy()
    groups = data["subject_id"].astype(str).to_numpy()
    folds = GroupKFold(n_splits=5)
    probability = np.full(len(data), np.nan)
    continuous, binary, categorical = identify_types_for_columns(data, predictors)
    for train_idx, test_idx in folds.split(data, y, groups):
        model = model_pipeline("Logistic Regression", continuous, binary, categorical)
        model.fit(data.iloc[train_idx][predictors], y[train_idx])
        probability[test_idx] = model.predict_proba(data.iloc[test_idx][predictors])[:, 1]
    if np.isnan(probability).any():
        raise AssertionError("Cross-fitted observability predictions are incomplete")
    performance = pd.DataFrame(
        [
            {
                "evaluable_definition": evaluable_column,
                **simple_metrics(y, probability),
                "five_fold_grouped_crossfit": True,
                "subject_overlap_across_fold_predictions": False,
            }
        ]
    )
    return probability, performance


def make_weights(evaluable: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    prevalence = float(np.mean(evaluable))
    bounded = np.clip(probability, 0.05, 0.95)
    raw = prevalence / bounded
    raw[~evaluable] = np.nan
    low, high = np.nanquantile(raw, [0.01, 0.99])
    clipped = np.clip(raw, low, high)
    audit = {
        "evaluable_prevalence": prevalence,
        "propensity_min": float(probability.min()),
        "propensity_p05": float(np.quantile(probability, 0.05)),
        "propensity_median": float(np.median(probability)),
        "propensity_p95": float(np.quantile(probability, 0.95)),
        "propensity_max": float(probability.max()),
        "weight_p01_before_final_clip": float(low),
        "weight_p99_before_final_clip": float(high),
        "weight_mean_evaluable": float(np.nanmean(clipped)),
        "effective_sample_size": float(np.nansum(clipped) ** 2 / np.nansum(clipped**2)),
    }
    return clipped, audit


def ipw_performance(
    onset: pd.DataFrame, predictions: pd.DataFrame, observability: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, weight_rows = [], []
    onset_index = onset.set_index("stay_id")
    for task, spec in TASKS.items():
        evaluable = bool_mask(onset[spec["evaluable"]]).to_numpy()
        weights, audit = make_weights(evaluable, observability[task])
        weight_rows.append({"task": task, **audit})
        weight_map = pd.Series(weights, index=onset["stay_id"])
        pred = predictions.loc[predictions["task"].eq(task)].copy()
        column = f"probability_{spec['selected_model'].lower().replace(' ', '_')}"
        pred["ipw"] = pred["stay_id"].map(weight_map)
        if pred["ipw"].isna().any():
            raise AssertionError(f"Missing IPW among evaluable held-out rows: {task}")
        y = pred["outcome"].astype(int).to_numpy()
        p = pred[column].to_numpy()
        rows.append({"task": task, "analysis": "complete case", **simple_metrics(y, p)})
        rows.append(
            {
                "task": task,
                "analysis": "stabilized IPW, propensity 0.05-0.95 and weight p1-p99",
                **simple_metrics(y, p, pred["ipw"].to_numpy()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def standardized_difference(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if not len(a) or not len(b):
        return np.nan
    denominator = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / denominator) if denominator > 0 else 0.0


def observability_comparison(onset: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "anchor_age", "charlson_score", "onset_aki_stage", "aki_onset_hours_from_icu",
        "baseline_scr_at_onset", "scr_n_icu_to_onset", "scr_n_prior_to_onset",
        "ckd", "cardiac_surgery", "non_cardiac_surgery",
    ]
    rows = []
    for task, spec in TASKS.items():
        evaluable = bool_mask(onset[spec["evaluable"]])
        for variable in variables:
            yes, no = onset.loc[evaluable, variable], onset.loc[~evaluable, variable]
            rows.append(
                {
                    "task": task,
                    "variable": variable,
                    "evaluable_n_nonmissing": int(yes.notna().sum()),
                    "nonevaluable_n_nonmissing": int(no.notna().sum()),
                    "evaluable_mean": pd.to_numeric(yes, errors="coerce").mean(),
                    "nonevaluable_mean": pd.to_numeric(no, errors="coerce").mean(),
                    "standardized_mean_difference": standardized_difference(yes, no),
                }
            )
        for careunit, group in onset.groupby(onset["first_careunit"].fillna("Unknown")):
            flag = onset["first_careunit"].fillna("Unknown").eq(careunit).astype(int)
            rows.append(
                {
                    "task": task,
                    "variable": f"first_careunit={careunit}",
                    "evaluable_n_nonmissing": int(evaluable.sum()),
                    "nonevaluable_n_nonmissing": int((~evaluable).sum()),
                    "evaluable_mean": flag[evaluable].mean(),
                    "nonevaluable_mean": flag[~evaluable].mean(),
                    "standardized_mean_difference": standardized_difference(flag[evaluable], flag[~evaluable]),
                }
            )
    return pd.DataFrame(rows)


def competing_events(onset: pd.DataFrame, cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = [
        "stay_id", "intime", "dischtime", "aki_onset_time_final", "end_observation_time",
        "hosp_death", "hospital_expire_flag", "los", "hosp_los",
    ]
    merged = onset.merge(cohort[[c for c in fields if c in cohort]], on="stay_id", how="left", validate="one_to_one")
    death = bool_mask(merged["hosp_death"]) | bool_mask(merged["hospital_expire_flag"])
    onset_time = pd.to_datetime(merged["aki_onset_time_final"], errors="coerce")
    discharge = pd.to_datetime(merged["dischtime"], errors="coerce")
    hours_to_discharge = (discharge - onset_time).dt.total_seconds() / 3600
    persistence_eval = bool_mask(merged["persistence_evaluable"])
    end_eval = bool_mask(merged["end_recovery_evaluable"])

    persistence_status = np.select(
        [
            persistence_eval,
            (~persistence_eval) & death & hours_to_discharge.le(48),
            (~persistence_eval) & (~death) & hours_to_discharge.le(48),
        ],
        ["renal outcome observed", "death within 48 h of AKI onset", "live discharge within 48 h of AKI onset"],
        default="insufficient SCr follow-up beyond 48 h",
    )
    end_status = np.select(
        [end_eval, (~end_eval) & death, (~end_eval) & (~death) & discharge.notna()],
        ["renal outcome observed", "in-hospital death before evaluable end SCr", "live discharge without evaluable end SCr"],
        default="hospitalized/unknown with insufficient end SCr",
    )
    merged["persistence_competing_status"] = persistence_status
    merged["end_recovery_competing_status"] = end_status
    rows = []
    for task, column in [
        ("persistent_aki", "persistence_competing_status"),
        ("nonrecovery", "end_recovery_competing_status"),
    ]:
        counts = merged[column].value_counts(dropna=False)
        for status, n in counts.items():
            rows.append({"task": task, "status": status, "n": int(n), "percent": 100 * n / len(merged)})
    return merged, pd.DataFrame(rows)


def composite_sensitivity(
    merged: pd.DataFrame, split: pd.DataFrame, predictors: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_map = split.drop_duplicates("subject_id").set_index("subject_id")["split"]
    merged["split"] = merged["subject_id"].map(split_map)
    persistence_eval = bool_mask(merged["persistence_evaluable"])
    end_eval = bool_mask(merged["end_recovery_evaluable"])
    merged["composite_persistent_or_early_death"] = np.nan
    merged.loc[persistence_eval, "composite_persistent_or_early_death"] = merged.loc[
        persistence_eval, "outcome_persistent_aki_beyond_48h"
    ].astype(int)
    early_death = merged["persistence_competing_status"].eq("death within 48 h of AKI onset")
    merged.loc[early_death, "composite_persistent_or_early_death"] = 1
    merged["composite_nonrecovery_or_inhospital_death"] = np.nan
    merged.loc[end_eval, "composite_nonrecovery_or_inhospital_death"] = merged.loc[
        end_eval, "outcome_not_recovered_at_end"
    ].astype(int)
    death_missing = merged["end_recovery_competing_status"].eq("in-hospital death before evaluable end SCr")
    merged.loc[death_missing, "composite_nonrecovery_or_inhospital_death"] = 1

    definitions = [
        ("persistent_aki", "composite_persistent_or_early_death", "XGBoost"),
        ("nonrecovery", "composite_nonrecovery_or_inhospital_death", "Logistic Regression"),
    ]
    perf, preds = [], []
    continuous, binary, categorical = identify_types_for_columns(merged, predictors)
    for task, outcome, model_name in definitions:
        data = merged.loc[merged[outcome].notna()].copy()
        train, test = data.loc[data["split"].eq("train")], data.loc[data["split"].eq("test")]
        model = model_pipeline(model_name, continuous, binary, categorical)
        model.fit(train[predictors], train[outcome].astype(int))
        probability = model.predict_proba(test[predictors])[:, 1]
        perf.append(
            {
                "task": task,
                "outcome": outcome,
                "model": model_name,
                "train_n": len(train),
                "train_event_n": int(train[outcome].sum()),
                "test_n": len(test),
                "test_event_n": int(test[outcome].sum()),
                **simple_metrics(test[outcome].astype(int).to_numpy(), probability),
            }
        )
        table = test[["subject_id", "hadm_id", "stay_id", outcome]].copy()
        table["task"] = task
        table["predicted_risk"] = probability
        preds.append(table)
    return pd.DataFrame(perf), pd.concat(preds, ignore_index=True)


def make_figure(
    onset: pd.DataFrame,
    observability: dict[str, np.ndarray],
    ipw: pd.DataFrame,
    competing: pd.DataFrame,
) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    colors = {"persistent_aki": "#4C78A8", "nonrecovery": "#D97706"}
    for task, spec in TASKS.items():
        y = bool_mask(onset[spec["evaluable"]]).astype(int)
        fpr, tpr, _ = roc_curve(y, observability[task])
        auc = roc_auc_score(y, observability[task])
        axes[0].plot(fpr, tpr, color=colors[task], linewidth=1.4, label=f"{spec['label']}: {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#98A2B3", linewidth=0.8)
    axes[0].set(xlabel="1 - specificity", ylabel="Sensitivity")
    axes[0].set_title("Outcome observability", loc="left", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=5.4)
    panel_label(axes[0], "a")

    metric = "auroc"
    x = np.arange(2)
    width = 0.34
    analysis_labels = {
        "complete case": "Complete case",
        "stabilized IPW, propensity 0.05-0.95 and weight p1-p99": "IPW-adjusted",
    }
    for i, analysis in enumerate(ipw["analysis"].unique()):
        values = [ipw.loc[(ipw["task"].eq(t)) & (ipw["analysis"].eq(analysis)), metric].iloc[0] for t in TASKS]
        axes[1].bar(x + (i - 0.5) * width, values, width, label=analysis_labels[analysis], color=["#98A2B3", "#7B61A8"][i])
    axes[1].set_xticks(x, ["Persistence", "Non-recovery"])
    axes[1].set_ylim(0.5, 0.9)
    axes[1].set_ylabel("Held-out AUROC")
    axes[1].set_title("Complete-case and IPW estimates", loc="left", fontweight="bold")
    axes[1].legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    panel_label(axes[1], "b")

    nonobserved = competing.loc[~competing["status"].eq("renal outcome observed")].copy()
    pivot = nonobserved.pivot(index="task", columns="status", values="percent").fillna(0)
    pivot = pivot.reindex(["persistent_aki", "nonrecovery"])
    bottom = np.zeros(len(pivot))
    status_style = {
        "death within 48 h of AKI onset": ("#B42318", "Death within 48 h"),
        "in-hospital death before evaluable end SCr": ("#D97706", "In-hospital death"),
        "live discharge within 48 h of AKI onset": ("#4C78A8", "Live discharge within 48 h"),
        "live discharge without evaluable end SCr": ("#7B61A8", "Live discharge without end SCr"),
        "insufficient SCr follow-up beyond 48 h": ("#72B7B2", "Insufficient SCr follow-up"),
        "hospitalized/unknown with insufficient end SCr": ("#B8B8B8", "Other insufficient end SCr"),
    }
    for status in pivot.columns:
        color, label = status_style[status]
        axes[2].bar(np.arange(len(pivot)), pivot[status], bottom=bottom, label=label, color=color)
        bottom += pivot[status].to_numpy()
    axes[2].set_xticks(np.arange(len(pivot)), ["Non-recovery" if x == "nonrecovery" else "Persistence" for x in pivot.index])
    axes[2].set_ylabel("AKI-onset cohort (%)")
    axes[2].set_title("Non-evaluable competing states", loc="left", fontweight="bold")
    axes[2].legend(fontsize=4.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    panel_label(axes[2], "c")
    save_figure(fig, "figure_v28_recovery_observability_ipw_competing")


def write_readme(
    onset: pd.DataFrame,
    obs_perf: pd.DataFrame,
    ipw: pd.DataFrame,
    competing: pd.DataFrame,
    composite: pd.DataFrame,
) -> None:
    lines = [
        "# v28 recovery observability, IPW, and competing-event sensitivity",
        "",
        "This post-lock secondary analysis retains the v26/v27 SCr phenotype definitions. It does not alter the primary incident-AKI analysis.",
        "",
        f"The AKI-onset cohort contained {len(onset):,} stays. Observability probabilities were estimated by five-fold subject-grouped cross-fitted logistic regression using only predictors available at AKI onset.",
        "",
        "Stabilized inverse-probability weights used the marginal evaluability prevalence, propensity truncation to 0.05-0.95, and final weight truncation at the 1st/99th percentiles. IPW estimates remain conditional on a missing-at-random assumption given measured onset features.",
        "",
        "Death and live discharge without the required follow-up SCr were reported as competing states. They were not coded as renal recovery. Exploratory adverse composites counted early/in-hospital death as adverse and excluded live-discharge cases with unknown renal status.",
        "",
        "## Key numerical results",
        "",
    ]
    for task, spec in TASKS.items():
        o = obs_perf.loc[obs_perf["evaluable_definition"].eq(spec["evaluable"]) & obs_perf["auroc"].notna()].iloc[0]
        cc = ipw.loc[ipw["task"].eq(task) & ipw["analysis"].eq("complete case")].iloc[0]
        wt = ipw.loc[ipw["task"].eq(task) & ipw["analysis"].str.startswith("stabilized")].iloc[0]
        lines.append(f"- {spec['label']}: evaluability {100*bool_mask(onset[spec['evaluable']]).mean():.1f}%; observability AUROC {o.auroc:.3f}; outcome-model AUROC {cc.auroc:.3f} complete case and {wt.auroc:.3f} IPW.")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "These analyses assess robustness to measured selection and competing events; they do not identify a causal recovery effect, solve informative monitoring under MNAR, or replace time-to-event competing-risk methods with exact death/discharge times.",
    ]
    (OUT / "audit_v28_recovery_readme.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    onset, cohort, predictions, split, predictors = load_inputs()
    observability, perf_blocks = {}, []
    for i, (task, spec) in enumerate(TASKS.items()):
        probability, block = crossfit_observability(onset, predictors, spec["evaluable"], RANDOM_STATE + i)
        observability[task] = probability
        perf_blocks.append(block.loc[block["auroc"].notna()].copy())
    obs_perf = pd.concat(perf_blocks, ignore_index=True)
    propensity = onset[["subject_id", "hadm_id", "stay_id"]].copy()
    for task, spec in TASKS.items():
        propensity[f"{task}_evaluable"] = bool_mask(onset[spec["evaluable"]])
        propensity[f"{task}_observability_probability_oof"] = observability[task]
    ipw, weight_audit = ipw_performance(onset, predictions, observability)
    comparison = observability_comparison(onset)
    merged, competing = competing_events(onset, cohort)
    composite_perf, composite_predictions = composite_sensitivity(merged, split, predictors)
    make_figure(onset, observability, ipw, competing)
    obs_perf.to_csv(OUT / "model_v28_observability_performance.csv", index=False)
    propensity.to_csv(OUT / "model_v28_observability_oof_predictions.csv", index=False)
    weight_audit.to_csv(OUT / "audit_v28_ipw_weight_distribution.csv", index=False)
    ipw.to_csv(OUT / "model_v28_recovery_ipw_performance.csv", index=False)
    comparison.to_csv(OUT / "audit_v28_evaluable_nonevaluable_comparison.csv", index=False)
    competing.to_csv(OUT / "audit_v28_recovery_competing_events.csv", index=False)
    merged[["subject_id", "hadm_id", "stay_id", "persistence_competing_status", "end_recovery_competing_status"]].to_csv(
        OUT / "cohort_v28_recovery_competing_status.csv", index=False
    )
    composite_perf.to_csv(OUT / "model_v28_adverse_composite_performance.csv", index=False)
    composite_predictions.to_csv(OUT / "model_v28_adverse_composite_test_predictions.csv", index=False)
    write_readme(onset, obs_perf, ipw, competing, composite_perf)
    print(f"Wrote v28 recovery selection analyses to {OUT}")


if __name__ == "__main__":
    main()
