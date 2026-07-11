"""v21 assumption-driven selection-bias sensitivity in eICU external validation.

This script has two explicitly secondary components:
1. Stabilized inverse-probability weighting (IPW) of outcome-observed external
   cases, using cross-fitted models for creatinine-record observability.
2. A pattern-mixture/tipping-point scenario for the unknown AKI risk among
   patients without adequate creatinine records.

Neither component identifies the true outcome distribution without assumptions.
They are sensitivity analyses, not corrections to the primary external result.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eicu_outcome_observability_selection_v18 import (  # noqa: E402
    CATEGORICAL_PREDICTORS, NUMERIC_PREDICTORS, make_pipeline, read_enrichment,
)

PROJECT_ROOT = SCRIPT_DIR.parent
V16_DIR = PROJECT_ROOT / "outputs" / "modeling_v16_eicu_external_validation"
EICU_ROOT = Path(os.environ.get("EICU_ROOT", str(PROJECT_ROOT.parents[1] / "eicu-collaborative-research-database-2.0")))
EICU_PATIENT_FILE = EICU_ROOT / "patient.csv" / "patient.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v21_eicu_selection_bias_sensitivity"
OUTCOME = "outcome_aki_after_landmark_to_7d"
LANDMARKS = (0, 6, 24)
HORIZON_HOURS = 168.0
RANDOM_STATE = 20250709

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def bool_series(x: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(x):
        return x.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce").fillna(0).astype(int).astype(bool)
    return x.astype("string").str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def build_selection_data() -> pd.DataFrame:
    cohort = pd.read_csv(V16_DIR / "cohort_v16_eicu_external_validation.csv.gz", low_memory=False).copy()
    cohort["analysis_evaluable"] = cohort["incident_aki_evaluable"].fillna(False).astype(int)
    cohort["creatinine_record_observable"] = (
        cohort["baseline_scr_source"].fillna("missing").ne("missing")
        & pd.to_numeric(cohort["post_index_scr_n_7d"], errors="coerce").fillna(0).gt(0)
    ).astype(int)
    patient, apache = read_enrichment(set(cohort["stay_id"].astype(int)))
    data = cohort.merge(patient, on="stay_id", how="left", validate="one_to_one").merge(apache, on="stay_id", how="left", validate="one_to_one")
    if data.duplicated("stay_id").any():
        raise ValueError("Selection data lost ICU-stay uniqueness")
    return data


def crossfit_observability(data: pd.DataFrame, include_hospital: bool) -> pd.Series:
    numeric = [x for x in NUMERIC_PREDICTORS if x in data]
    categorical = [x for x in CATEGORICAL_PREDICTORS if x in data]
    if include_hospital:
        categorical.append("hospitalid")
    y = data["creatinine_record_observable"].astype(int).to_numpy()
    groups = data["subject_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    estimated = np.full(len(data), np.nan)
    for train_idx, test_idx in splitter.split(data, y, groups):
        pipe = make_pipeline(numeric, categorical)
        pipe.fit(data.iloc[train_idx][[*numeric, *categorical]], y[train_idx])
        estimated[test_idx] = pipe.predict_proba(data.iloc[test_idx][[*numeric, *categorical]])[:, 1]
    if np.isnan(estimated).any():
        raise RuntimeError("Cross-fitted observability probabilities incomplete")
    return pd.Series(estimated, index=data.index)


def weighted_calibration(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logit(p).reshape(-1, 1), y, sample_weight=w)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def weighted_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> dict[str, float]:
    if w is None:
        w = np.ones(len(y), dtype=float)
    y, p, w = np.asarray(y, dtype=int), np.asarray(p, dtype=float), np.asarray(w, dtype=float)
    intercept, slope = weighted_calibration(y, p, w)
    return {
        "n": int(len(y)), "event_n": int(y.sum()), "weighted_event_rate": float(np.average(y, weights=w)),
        "auroc": float(roc_auc_score(y, p, sample_weight=w)), "auprc": float(average_precision_score(y, p, sample_weight=w)),
        "brier_score": float(brier_score_loss(y, p, sample_weight=w)), "calibration_intercept": intercept,
        "calibration_slope": slope, "weight_effective_sample_size": float((w.sum() ** 2) / np.sum(w ** 2)),
    }


def active_competing_prediction_rows(predictions: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    patient = pd.read_csv(EICU_PATIENT_FILE, usecols=["patientunitstayid", "unitdischargeoffset"], low_memory=False).rename(columns={"patientunitstayid": "stay_id"})
    patient["stay_id"] = pd.to_numeric(patient["stay_id"], errors="coerce").astype(int)
    patient["unit_exit_hours"] = pd.to_numeric(patient["unitdischargeoffset"], errors="coerce") / 60
    # Keep the fitted selection probabilities and stabilized weights when attaching
    # the complete-case prediction records.  They are calculated upstream in
    # ``ipw_performance`` and must remain aligned by stay_id.
    cols = [
        "stay_id", "creatinine_record_observable", "analysis_evaluable", "aki_final", "aki_onset_hours_final",
        "p_observable_clinical", "p_observable_clinical_hospital",
        "weight_ipw_clinical", "weight_ipw_clinical_hospital",
    ]
    merged = predictions.merge(selection[cols], on="stay_id", how="left", validate="many_to_one").merge(patient[["stay_id", "unit_exit_hours"]], on="stay_id", how="left", validate="many_to_one")
    rows = []
    for landmark in LANDMARKS:
        d = merged.loc[merged["landmark_hours"].eq(landmark)].copy()
        d = d.loc[pd.to_numeric(d["unit_exit_hours"], errors="coerce").gt(landmark)].copy()
        onset = pd.to_numeric(d["aki_onset_hours_final"], errors="coerce")
        exit_time = pd.to_numeric(d["unit_exit_hours"], errors="coerce")
        d["competing_aki_before_unit_exit"] = (
            pd.to_numeric(d["aki_final"], errors="coerce").fillna(0).eq(1)
            & onset.gt(landmark) & onset.le(exit_time) & onset.le(HORIZON_HOURS)
        ).astype(int)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def ipw_performance(selection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical_p = crossfit_observability(selection, include_hospital=False)
    hospital_p = crossfit_observability(selection, include_hospital=True)
    selection = selection.copy()
    selection["p_observable_clinical"] = clinical_p
    selection["p_observable_clinical_hospital"] = hospital_p
    pi = float(selection["creatinine_record_observable"].mean())
    diag_rows = []
    for name, column in [("ipw_clinical", "p_observable_clinical"), ("ipw_clinical_hospital", "p_observable_clinical_hospital")]:
        p = selection[column].clip(.05, .95)
        raw = pi / p
        observed_raw = raw.loc[selection["analysis_evaluable"].eq(1)]
        low, high = observed_raw.quantile(.01), observed_raw.quantile(.99)
        selection[f"weight_{name}"] = raw.clip(low, high)
        diag_rows.append({
            "weight_model": name, "selection_outcome": "creatinine_record_observable", "stabilized_numerator": pi,
            "propensity_min": float(p.min()), "propensity_max": float(p.max()), "propensity_lt_0_05_n": int((selection[column] < .05).sum()),
            "propensity_gt_0_95_n": int((selection[column] > .95).sum()), "weight_p01": float(low), "weight_p99": float(high),
            "observed_weight_mean": float(selection.loc[selection["analysis_evaluable"].eq(1), f"weight_{name}"].mean()),
            "observed_weight_effective_sample_size": float((selection.loc[selection["analysis_evaluable"].eq(1), f"weight_{name}"].sum() ** 2) / np.sum(selection.loc[selection["analysis_evaluable"].eq(1), f"weight_{name}"] ** 2)),
        })
    predictions = pd.read_csv(V16_DIR / "model_v16_portable_external_predictions.csv", low_memory=False)
    active = active_competing_prediction_rows(predictions, selection)
    rows = []
    for landmark in LANDMARKS:
        d = active.loc[active["landmark_hours"].eq(landmark) & active["analysis_evaluable"].eq(1)].copy()
        y = d["competing_aki_before_unit_exit"].astype(int).to_numpy()
        p = d["predicted_risk_portable_model"].to_numpy()
        for method, weights in [("unweighted_complete_case", np.ones(len(d))), ("ipw_clinical", d["weight_ipw_clinical"].to_numpy()), ("ipw_clinical_hospital", d["weight_ipw_clinical_hospital"].to_numpy())]:
            row = {"landmark_hours": landmark, "method": method, "risk_definition": "active ICU, AKI before unit exit", "selection_assumption": "MAR conditional on model covariates" if method != "unweighted_complete_case" else "none"}
            row.update(weighted_metrics(y, p, weights))
            rows.append(row)
    selection_out = selection[["stay_id", "creatinine_record_observable", "analysis_evaluable", "p_observable_clinical", "p_observable_clinical_hospital", "weight_ipw_clinical", "weight_ipw_clinical_hospital"]].copy()
    return pd.DataFrame(rows), pd.DataFrame(diag_rows), selection_out


def pattern_mixture(selection: pd.DataFrame) -> pd.DataFrame:
    known_preexisting = bool_series(selection["preexisting_aki_at_or_before_index"]) if "preexisting_aki_at_or_before_index" in selection else pd.Series(False, index=selection.index)
    target = selection.loc[~known_preexisting].copy()
    observed = target.loc[target["analysis_evaluable"].eq(1)]
    unavailable = target.loc[target["analysis_evaluable"].eq(0)]
    observed_risk = float(pd.to_numeric(observed["aki_final"], errors="coerce").mean())
    rows = []
    for rr in [.50, .75, 1.00, 1.25, 1.50, 2.00]:
        missing_risk = min(observed_risk * rr, .99)
        overall = (len(observed) * observed_risk + len(unavailable) * missing_risk) / len(target)
        rows.append({
            "assumed_unobserved_to_observed_aki_risk_ratio": rr, "observed_n": len(observed), "unobserved_n": len(unavailable),
            "analysis_target_n_excluding_known_preindex_aki": len(target), "observed_aki_risk": observed_risk,
            "assumed_unobserved_aki_risk": missing_risk, "implied_population_aki_incidence": overall,
            "assumption_note": "Pattern-mixture scenario; no outcome is imputed and no causal/missing-at-random claim is made.",
        })
    return pd.DataFrame(rows)


def make_figures(ipw: pd.DataFrame, scenario: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"unweighted_complete_case": "#4c78a8", "ipw_clinical": "#59a14f", "ipw_clinical_hospital": "#e17c05"}
    labels = {"unweighted_complete_case": "Unweighted complete cases", "ipw_clinical": "IPW: clinical", "ipw_clinical_hospital": "IPW: clinical + hospital"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), dpi=180, sharey=True)
    for ax, landmark in zip(axes, LANDMARKS):
        d = ipw.loc[ipw["landmark_hours"].eq(landmark)]
        x = np.arange(len(d))
        ax.scatter(x, d["auroc"], s=65, c=[colors[m] for m in d["method"]])
        for i, row in enumerate(d.itertuples()):
            ax.text(i, row.auroc + .003, f"{row.auroc:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x, ["Complete\ncase", "IPW\nclinical", "IPW\nclinical+\nhospital"])
        ax.set_ylim(.60, .78); ax.set_title(f"{landmark} h")
    axes[0].set_ylabel("Weighted AUROC")
    fig.suptitle("IPW sensitivity to creatinine-record selection", y=1.03)
    fig.text(.5, .965, "Active ICU risk set; AKI before unit exit; IPW assumes MAR conditional on observed covariates", ha="center", fontsize=8.5, color="#5f6675")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUTPUT_DIR / "figure_v21_ipw_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    ax.plot(scenario["assumed_unobserved_to_observed_aki_risk_ratio"], scenario["implied_population_aki_incidence"] * 100, marker="o", color="#e17c05")
    ax.axvline(1, color="#687080", linestyle="--", linewidth=1)
    ax.set_xlabel("Assumed AKI risk ratio: unobserved vs observed patients")
    ax.set_ylabel("Implied population 7-day AKI incidence (%)")
    ax.set_title("Pattern-mixture selection-bias scenario")
    fig.text(.5, .01, "Excludes known pre-index AKI; scenario varies unobserved outcome risk without imputing individual outcomes", ha="center", fontsize=8.5, color="#5f6675")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUTPUT_DIR / "figure_v21_pattern_mixture_aki_incidence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_readme(ipw: pd.DataFrame, diagnostics: pd.DataFrame, scenario: pd.DataFrame) -> None:
    lines = ["# v21 assumption-driven eICU selection-bias sensitivity", "", "## Scope", ""]
    lines.append("This analysis does not recover unobserved AKI outcomes. It quantifies how external performance and population incidence would change under explicit, unverifiable assumptions about creatinine-record observability.")
    lines.extend(["", "## IPW sensitivity", ""])
    lines.append("Observability probabilities were estimated by five-fold subject-grouped cross-fitting. Stabilized weights were inverse probabilities of creatinine-record observability, truncated at the 1st and 99th percentiles among analysis-evaluable stays. IPW estimates assume missing at random conditional on included covariates; this assumption is not testable from these data.")
    for r in ipw.itertuples():
        lines.append(f"- {int(r.landmark_hours)} h, {r.method}: weighted AUROC {r.auroc:.3f}, AUPRC {r.auprc:.3f}, Brier {r.brier_score:.3f}, weighted event rate {r.weighted_event_rate*100:.1f}%, ESS {r.weight_effective_sample_size:.0f}.")
    lines.extend(["", "## Pattern-mixture scenarios", ""])
    for r in scenario.itertuples():
        lines.append(f"- If unobserved AKI risk is {r.assumed_unobserved_to_observed_aki_risk_ratio:.2f}× observed risk, implied population 7-day AKI incidence is {r.implied_population_aki_incidence*100:.1f}% (observed risk {r.observed_aki_risk*100:.1f}%; assumed unobserved risk {r.assumed_unobserved_aki_risk*100:.1f}%).")
    lines.extend(["", "## Required interpretation", "", "IPW and scenario results are conditional on assumptions, not evidence that selection bias has been eliminated. They should be reported in supplementary material as robustness bounds. The primary external results remain complete-case, feature-harmonized validation with explicit observability limitations."])
    (OUTPUT_DIR / "audit_v21_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selection = build_selection_data()
    ipw, diagnostics, selection_scores = ipw_performance(selection)
    scenario = pattern_mixture(selection)
    ipw.to_csv(OUTPUT_DIR / "analysis_v21_ipw_weighted_external_performance.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "audit_v21_ipw_propensity_diagnostics.csv", index=False)
    selection_scores.to_csv(OUTPUT_DIR / "analysis_v21_crossfitted_observability_scores.csv", index=False)
    scenario.to_csv(OUTPUT_DIR / "analysis_v21_pattern_mixture_aki_incidence.csv", index=False)
    make_figures(ipw, scenario)
    write_readme(ipw, diagnostics, scenario)
    print(f"Wrote v21 outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
