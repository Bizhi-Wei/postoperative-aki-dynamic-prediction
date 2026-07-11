"""v17 eICU recalibration, hospital heterogeneity, and baseline sensitivity.

Uses only frozen v16 eICU predictions.  The primary MIMIC models and their
predictions are not refit. Formal external recalibration is evaluated with an
80/20 *hospital-grouped* split: calibration parameters are learned in 80% of
hospitals and assessed in held-out hospitals.
"""

from __future__ import annotations

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
from sklearn.model_selection import GroupShuffleSplit

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recalibration_and_measurement_intensity_v13 import calibration_intercept_slope  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = SCRIPT_DIR.parent
V16_DIR = PROJECT_ROOT / "outputs" / "modeling_v16_eicu_external_validation"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v17_eicu_recalibration_heterogeneity"
RANDOM_STATE = 20250709
OUTCOME = "outcome_aki_after_landmark_to_7d"


def logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -30, 30)))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "n": int(len(y)), "event_n": int(y.sum()), "event_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)), "mean_predicted_risk": float(p.mean()),
        "observed_expected_ratio": float(y.mean() / p.mean()) if p.mean() else np.nan,
        "calibration_intercept": intercept, "calibration_slope": slope,
    }


def bootstrap_auroc(y: np.ndarray, p: np.ndarray, draws: int = 500, seed_offset: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE + seed_offset)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            values.append(roc_auc_score(y[idx], p[idx]))
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def select_hospital_split(predictions: pd.DataFrame) -> tuple[set[int], set[int], pd.DataFrame]:
    """Select a stable 80/20 hospital split with approximate outcome balance."""
    base = predictions.loc[predictions["landmark_hours"].eq(0)].copy()
    groups = pd.to_numeric(base["hospitalid"], errors="coerce").astype(int)
    y = base[OUTCOME].astype(int)
    overall_rate = float(y.mean())
    splitter = GroupShuffleSplit(n_splits=500, test_size=.20, random_state=RANDOM_STATE)
    best = None
    for calibration_idx, test_idx in splitter.split(base, y, groups):
        test_rate = float(y.iloc[test_idx].mean())
        score = abs(len(test_idx) / len(base) - .20) + abs(test_rate - overall_rate)
        if best is None or score < best[0]:
            best = (score, calibration_idx, test_idx)
    assert best is not None
    _, calibration_idx, test_idx = best
    calibration_hospitals = set(groups.iloc[calibration_idx].astype(int))
    test_hospitals = set(groups.iloc[test_idx].astype(int))
    if calibration_hospitals & test_hospitals:
        raise AssertionError("Hospital overlap between recalibration and validation partitions")
    audit = pd.DataFrame([
        {"partition": "calibration hospitals", "hospital_n": len(calibration_hospitals), "n_0h": len(calibration_idx), "event_rate_0h": float(y.iloc[calibration_idx].mean())},
        {"partition": "held-out hospitals", "hospital_n": len(test_hospitals), "n_0h": len(test_idx), "event_rate_0h": float(y.iloc[test_idx].mean())},
    ])
    return calibration_hospitals, test_hospitals, audit


def fit_intercept_only(y: np.ndarray, original_probability: np.ndarray) -> float:
    """Maximum-likelihood calibration-in-the-large with slope fixed at one."""
    z = logit(original_probability)
    intercept = 0.0
    for _ in range(100):
        fitted = expit(z + intercept)
        gradient = float(np.sum(fitted - y))
        hessian = float(np.sum(fitted * (1 - fitted)))
        update = gradient / max(hessian, 1e-10)
        intercept -= update
        if abs(update) < 1e-10:
            break
    return float(intercept)


def calculate_recalibration(predictions: pd.DataFrame, calibration_hospitals: set[int], test_hospitals: set[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results, parameters, test_outputs = [], [], []
    for landmark in (0, 6, 24):
        data = predictions.loc[predictions["landmark_hours"].eq(landmark)].copy()
        data["hospitalid"] = pd.to_numeric(data["hospitalid"], errors="coerce").astype(int)
        calibration = data.loc[data["hospitalid"].isin(calibration_hospitals)].copy()
        test = data.loc[data["hospitalid"].isin(test_hospitals)].copy()
        y_cal = calibration[OUTCOME].astype(int).to_numpy()
        p_cal = calibration["predicted_risk_portable_model"].to_numpy()
        y_test = test[OUTCOME].astype(int).to_numpy()
        p_test = test["predicted_risk_portable_model"].to_numpy()
        offset = fit_intercept_only(y_cal, p_cal)
        platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        platt.fit(logit(p_cal).reshape(-1, 1), y_cal)
        platt_intercept = float(platt.intercept_[0])
        platt_slope = float(platt.coef_[0, 0])
        p_intercept = expit(logit(p_test) + offset)
        p_logistic = expit(platt_intercept + platt_slope * logit(p_test))
        for method, probabilities in [
            ("frozen_unrecalibrated", p_test),
            ("intercept_only_update", p_intercept),
            ("logistic_recalibration_update", p_logistic),
        ]:
            row = {"landmark_hours": landmark, "evaluation_partition": "held-out eICU hospitals", "method": method}
            row.update(metrics(y_test, probabilities))
            low, high = bootstrap_auroc(y_test, probabilities, seed_offset=landmark * 100 + len(method))
            row.update({"auroc_ci_lower": low, "auroc_ci_upper": high})
            results.append(row)
        parameters.append({
            "landmark_hours": landmark, "calibration_hospital_n": calibration["hospitalid"].nunique(),
            "calibration_n": len(calibration), "calibration_event_n": int(y_cal.sum()),
            "intercept_only_offset": offset, "logistic_recalibration_intercept": platt_intercept,
            "logistic_recalibration_slope": platt_slope,
            "parameter_note": "Learned using calibration hospitals only; applied unchanged to held-out hospitals.",
        })
        out = test[["stay_id", "subject_id", "hospitalid", "operative_system", "baseline_scr_source", OUTCOME]].copy()
        out["landmark_hours"] = landmark
        out["predicted_risk_frozen"] = p_test
        out["predicted_risk_intercept_updated"] = p_intercept
        out["predicted_risk_logistic_recalibrated"] = p_logistic
        test_outputs.append(out)
    return pd.DataFrame(results), pd.DataFrame(parameters), pd.concat(test_outputs, ignore_index=True)


def hospital_heterogeneity(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, summary_rows = [], []
    for landmark in (0, 6, 24):
        data = predictions.loc[predictions["landmark_hours"].eq(landmark)].copy()
        for hospital, group in data.groupby("hospitalid", sort=True):
            y = group[OUTCOME].astype(int).to_numpy()
            p = group["predicted_risk_portable_model"].to_numpy()
            row = {"landmark_hours": landmark, "hospitalid": hospital}
            row.update(metrics(y, p) if y.sum() >= 1 and (len(y) - y.sum()) >= 1 else {"n": len(y), "event_n": int(y.sum()), "event_rate": float(y.mean()), "auroc": np.nan, "auprc": np.nan, "brier_score": np.nan, "mean_predicted_risk": float(p.mean()), "observed_expected_ratio": float(y.mean()/p.mean()) if p.mean() else np.nan, "calibration_intercept": np.nan, "calibration_slope": np.nan})
            row["auroc_evaluable"] = bool(len(group) >= 100 and y.sum() >= 20 and (len(y) - y.sum()) >= 20)
            row["calibration_evaluable"] = bool(len(group) >= 200 and y.sum() >= 30 and (len(y) - y.sum()) >= 30)
            rows.append(row)
        table = pd.DataFrame([r for r in rows if r["landmark_hours"] == landmark])
        aucs = table.loc[table["auroc_evaluable"], "auroc"].dropna()
        slopes = table.loc[table["calibration_evaluable"], "calibration_slope"].dropna()
        summary_rows.append({
            "landmark_hours": landmark, "hospital_n_with_patients": int(table["hospitalid"].nunique()),
            "hospital_n_auroc_evaluable": int(len(aucs)), "hospital_auroc_median": float(aucs.median()) if len(aucs) else np.nan,
            "hospital_auroc_iqr_lower": float(aucs.quantile(.25)) if len(aucs) else np.nan,
            "hospital_auroc_iqr_upper": float(aucs.quantile(.75)) if len(aucs) else np.nan,
            "hospital_n_calibration_evaluable": int(len(slopes)), "hospital_calibration_slope_median": float(slopes.median()) if len(slopes) else np.nan,
            "hospital_calibration_slope_iqr_lower": float(slopes.quantile(.25)) if len(slopes) else np.nan,
            "hospital_calibration_slope_iqr_upper": float(slopes.quantile(.75)) if len(slopes) else np.nan,
        })
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def strict_baseline_sensitivity(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for landmark in (0, 6, 24):
        full = predictions.loc[predictions["landmark_hours"].eq(landmark)].copy()
        strict = full.loc[full["baseline_scr_source"].eq("lowest_scr_7d_pre_icu")].copy()
        for cohort, data in [("all_eicu_evaluable", full), ("strict_pre_icu_baseline_only", strict)]:
            y = data[OUTCOME].astype(int).to_numpy()
            p = data["predicted_risk_portable_model"].to_numpy()
            row = {"landmark_hours": landmark, "sensitivity_cohort": cohort}
            row.update(metrics(y, p))
            low, high = bootstrap_auroc(y, p, seed_offset=1000 + landmark)
            row.update({"auroc_ci_lower": low, "auroc_ci_upper": high, "cohort_percent_of_all_evaluable": float(100 * len(data) / len(full))})
            rows.append(row)
    return pd.DataFrame(rows)


def make_figures(recalibration: pd.DataFrame, recalibrated_predictions: pd.DataFrame, hospitals: pd.DataFrame, baseline: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"frozen_unrecalibrated": "#4c78a8", "intercept_only_update": "#59a14f", "logistic_recalibration_update": "#e17c05"}
    labels = {"frozen_unrecalibrated": "Frozen", "intercept_only_update": "Intercept update", "logistic_recalibration_update": "Logistic recalibration"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), dpi=180, sharey=True)
    for ax, landmark in zip(axes, (0, 6, 24)):
        data = recalibrated_predictions.loc[recalibrated_predictions["landmark_hours"].eq(landmark)]
        for method, col in [("frozen_unrecalibrated", "predicted_risk_frozen"), ("intercept_only_update", "predicted_risk_intercept_updated"), ("logistic_recalibration_update", "predicted_risk_logistic_recalibrated")]:
            work = data[[OUTCOME, col]].copy()
            work["bin"] = pd.qcut(work[col], q=min(10, work[col].nunique()), duplicates="drop")
            bins = work.groupby("bin", observed=False).agg(pred=(col, "mean"), obs=(OUTCOME, "mean"))
            ax.plot(bins["pred"], bins["obs"], marker="o", linewidth=1.6, color=colors[method], label=labels[method])
        ax.plot([0, 1], [0, 1], "--", color="#687080", linewidth=1)
        ax.set_title(f"{landmark} h")
        ax.set_xlabel("Mean predicted risk")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed AKI risk")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Hospital-held-out external recalibration", y=1.03)
    fig.text(.5, .965, "Parameters learned in 80% of hospitals; calibration curves evaluated in distinct hospitals", ha="center", fontsize=9, color="#5f6675")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUTPUT_DIR / "figure_v17_external_recalibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), dpi=180, sharey=True)
    for ax, landmark in zip(axes, (0, 6, 24)):
        data = hospitals.loc[(hospitals["landmark_hours"] == landmark) & hospitals["auroc_evaluable"]].copy()
        sizes = np.clip(data["n"] / max(data["n"].max(), 1) * 140, 25, 140)
        sc = ax.scatter(data["event_rate"] * 100, data["auroc"], s=sizes, color="#4c78a8", alpha=.75, edgecolor="white", linewidth=.5)
        ax.axhline(.5, color="#687080", linestyle="--", linewidth=1)
        ax.set_title(f"{landmark} h (n hospitals={len(data)})")
        ax.set_xlabel("Hospital AKI incidence (%)")
        ax.set_ylim(.45, .90)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("Hospital-specific AUROC")
    fig.suptitle("Hospital-level heterogeneity of frozen-model discrimination", y=1.03)
    fig.text(.5, .965, "Hospitals with ≥100 patients and ≥20 AKI events; point area proportional to sample size", ha="center", fontsize=9, color="#5f6675")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUTPUT_DIR / "figure_v17_hospital_heterogeneity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    plot = baseline.copy()
    x = np.arange(3)
    width = .12
    for j, cohort in enumerate(["all_eicu_evaluable", "strict_pre_icu_baseline_only"]):
        d = plot.loc[plot["sensitivity_cohort"].eq(cohort)].sort_values("landmark_hours")
        xpos = x + (j - .5) * width
        color = "#4c78a8" if j == 0 else "#e17c05"
        ax.errorbar(xpos, d["auroc"], yerr=[d["auroc"] - d["auroc_ci_lower"], d["auroc_ci_upper"] - d["auroc"]], fmt="o", markersize=7, color=color, ecolor=color, capsize=3, label="All evaluable" if j == 0 else "Strict pre-ICU baseline only")
    ax.set_xlim(-.35, 2.35)
    ax.set_xticks(x)
    ax.set_xticklabels(["0 h", "6 h", "24 h"])
    ax.set_ylim(.64, .71)
    ax.set_ylabel("AUROC")
    ax.legend(frameon=False)
    fig.suptitle("Sensitivity to strict pre-ICU baseline creatinine", y=.97)
    fig.text(.5, .045, "Frozen portable models; 95% bootstrap confidence intervals; focused AUROC scale (0.64–0.71)", ha="center", fontsize=9, color="#5f6675")
    fig.subplots_adjust(left=.12, right=.97, bottom=.16, top=.89)
    fig.savefig(OUTPUT_DIR / "figure_v17_strict_baseline_sensitivity.png", dpi=300)
    plt.close(fig)


def write_readme(split: pd.DataFrame, recalibration: pd.DataFrame, parameters: pd.DataFrame, hospital_summary: pd.DataFrame, baseline: pd.DataFrame) -> None:
    lines = ["# v17 eICU recalibration, hospital heterogeneity, and baseline sensitivity", "", "## External recalibration design", ""]
    lines.append("Frozen MIMIC-trained portable-model predictions from v16 were used. eICU hospitals, rather than individual stays, were split 80/20. Calibration update parameters were learned in the 80% hospital partition and evaluated without refitting in the held-out 20% hospital partition.")
    lines.append("")
    lines.append("Two updates were assessed: a calibration-in-the-large intercept update and logistic recalibration of the original prediction logit. The primary external-validation result remains the frozen, unrecalibrated model.")
    lines.extend(["", "## Held-out hospital recalibration results", ""])
    for landmark in (0, 6, 24):
        d = recalibration.loc[recalibration["landmark_hours"].eq(landmark)]
        raw = d.loc[d["method"].eq("frozen_unrecalibrated")].iloc[0]
        updated = d.loc[d["method"].eq("logistic_recalibration_update")].iloc[0]
        param = parameters.loc[parameters["landmark_hours"].eq(landmark)].iloc[0]
        lines.append(f"- {landmark} h: held-out hospitals n={int(raw.n):,}; frozen Brier {raw.brier_score:.3f}, slope {raw.calibration_slope:.2f}; logistic recalibration Brier {updated.brier_score:.3f}, slope {updated.calibration_slope:.2f}; learned intercept {param.logistic_recalibration_intercept:.3f}, slope {param.logistic_recalibration_slope:.3f}.")
    lines.extend(["", "## Hospital heterogeneity", ""])
    for r in hospital_summary.itertuples():
        lines.append(f"- {int(r.landmark_hours)} h: {int(r.hospital_n_auroc_evaluable)} hospitals met AUROC precision criteria; median hospital AUROC {r.hospital_auroc_median:.3f} (IQR {r.hospital_auroc_iqr_lower:.3f}–{r.hospital_auroc_iqr_upper:.3f}).")
    lines.extend(["", "## Strict pre-ICU baseline sensitivity", ""])
    for r in baseline.loc[baseline["sensitivity_cohort"].eq("strict_pre_icu_baseline_only")].itertuples():
        lines.append(f"- {int(r.landmark_hours)} h: n={int(r.n):,} ({r.cohort_percent_of_all_evaluable:.1f}% of evaluable external cohort), AUROC {r.auroc:.3f} (95% CI {r.auroc_ci_lower:.3f}–{r.auroc_ci_upper:.3f}).")
    lines.extend(["", "## Interpretation", "", "Hospital-level updating can improve calibration at the receiving system, but it is a local adaptation rather than independent validation. The 24-h model should not be deployed in a new hospital without recalibration and prospective monitoring. Hospital-level discrimination and calibration variation supports reporting transportability as an empirical, not assumed, property."])
    (OUTPUT_DIR / "audit_v17_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(V16_DIR / "model_v16_portable_external_predictions.csv", low_memory=False)
    cohort = pd.read_csv(V16_DIR / "cohort_v16_eicu_external_validation.csv.gz", low_memory=False, usecols=["stay_id", "baseline_scr_source"])
    if preds.duplicated(["stay_id", "landmark_hours"]).any():
        raise ValueError("v16 external predictions are not unique at stay/landmark grain")
    preds = preds.merge(cohort, on="stay_id", how="left", validate="many_to_one")
    if preds["baseline_scr_source"].isna().any():
        raise ValueError("Baseline-source mapping failed for some external predictions")
    cal_hospitals, test_hospitals, split = select_hospital_split(preds)
    recalibration, parameters, recalibrated_preds = calculate_recalibration(preds, cal_hospitals, test_hospitals)
    hospitals, hospital_summary = hospital_heterogeneity(preds)
    baseline = strict_baseline_sensitivity(preds)
    split.to_csv(OUTPUT_DIR / "audit_v17_hospital_recalibration_split.csv", index=False)
    recalibration.to_csv(OUTPUT_DIR / "model_v17_external_recalibration_performance.csv", index=False)
    parameters.to_csv(OUTPUT_DIR / "model_v17_external_recalibration_parameters.csv", index=False)
    recalibrated_preds.to_csv(OUTPUT_DIR / "model_v17_heldout_hospital_recalibrated_predictions.csv", index=False)
    hospitals.to_csv(OUTPUT_DIR / "analysis_v17_hospital_heterogeneity.csv", index=False)
    hospital_summary.to_csv(OUTPUT_DIR / "analysis_v17_hospital_heterogeneity_summary.csv", index=False)
    baseline.to_csv(OUTPUT_DIR / "analysis_v17_strict_pre_icu_baseline_sensitivity.csv", index=False)
    make_figures(recalibration, recalibrated_preds, hospitals, baseline)
    write_readme(split, recalibration, parameters, hospital_summary, baseline)
    print(f"Wrote v17 outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
