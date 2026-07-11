"""v18 eICU outcome-observability and measurement-selection analysis.

This analysis audits why postoperative eICU stays were or were not evaluable
for the externally validated SCr-KDIGO AKI outcome.  It does not impute an AKI
outcome for unevaluable stays and does not modify any prediction-model result.

Observability is defined as a documented baseline creatinine, no pre-index AKI,
and at least one post-index creatinine within 7 days, exactly as in v16.
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

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
V16_DIR = PROJECT_ROOT / "outputs" / "modeling_v16_eicu_external_validation"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v18_eicu_outcome_observability"
EICU_ROOT = Path(os.environ.get("EICU_ROOT", str(PROJECT_ROOT.parents[1] / "eicu-collaborative-research-database-2.0")))
PATIENT_FILE = EICU_ROOT / "patient.csv" / "patient.csv"
APACHE_RESULT_FILE = EICU_ROOT / "apachePatientResult.csv" / "apachePatientResult.csv"
RANDOM_STATE = 20250709

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

COMORBIDITIES = ["chf", "hypertension", "dm", "ckd", "copd", "liver", "cancer", "pvd", "stroke", "mi", "obesity", "anemia"]
NUMERIC_PREDICTORS = ["anchor_age", *COMORBIDITIES, "charlson_score", "acutephysiologyscore", "apachescore", "predictedicumortality", "predictedhospitalmortality"]
CATEGORICAL_PREDICTORS = ["gender", "operative_system", "unittype"]


def bool_like(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower().isin(["expired", "death", "dead", "true", "1", "yes"]).astype(int)


def read_enrichment(stay_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient = pd.read_csv(
        PATIENT_FILE,
        usecols=["patientunitstayid", "unitdischargestatus", "hospitaldischargestatus", "unitdischargeoffset", "hospitaldischargeoffset", "unitadmitsource", "hospitaladmitsource"],
        low_memory=False,
    )
    patient = patient.loc[pd.to_numeric(patient["patientunitstayid"], errors="coerce").isin(stay_ids)].copy()
    patient = patient.rename(columns={"patientunitstayid": "stay_id"})
    patient["stay_id"] = pd.to_numeric(patient["stay_id"], errors="coerce").astype(int)
    if patient.duplicated("stay_id").any():
        raise ValueError("eICU patient table should have exactly one row per ICU stay")
    patient["icu_death"] = bool_like(patient["unitdischargestatus"])
    patient["hospital_death"] = bool_like(patient["hospitaldischargestatus"])

    apache = pd.read_csv(
        APACHE_RESULT_FILE,
        usecols=["patientunitstayid", "apacheversion", "acutephysiologyscore", "apachescore", "predictedicumortality", "actualicumortality", "predictedhospitalmortality", "actualhospitalmortality"],
        low_memory=False,
    )
    apache = apache.loc[pd.to_numeric(apache["patientunitstayid"], errors="coerce").isin(stay_ids)].copy()
    # Each eICU stay has IV and IVa rows.  IVa is the fixed primary version.
    apache = apache.loc[apache["apacheversion"].eq("IVa")].copy()
    apache = apache.rename(columns={"patientunitstayid": "stay_id"})
    apache["stay_id"] = pd.to_numeric(apache["stay_id"], errors="coerce").astype(int)
    if apache.duplicated("stay_id").any():
        raise ValueError("APACHE IVa table is not unique at ICU-stay grain")
    for col in ["acutephysiologyscore", "apachescore", "predictedicumortality", "predictedhospitalmortality"]:
        apache[col] = pd.to_numeric(apache[col], errors="coerce")
    return patient, apache


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def group_comparison(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    evaluable = data.loc[data["outcome_evaluable"].eq(1)].copy()
    unevaluable = data.loc[data["outcome_evaluable"].eq(0)].copy()
    continuous = ["anchor_age", "charlson_score", "acutephysiologyscore", "apachescore", "predictedicumortality", "predictedhospitalmortality", "unitdischargeoffset", "hospitaldischargeoffset"]
    binary = [*COMORBIDITIES, "icu_death", "hospital_death"]
    categorical = ["operative_system", "unittype", "gender", "unitadmitsource", "hospitaladmitsource", "ineligibility_reason"]
    for var in continuous:
        if var not in data:
            continue
        eva, uneva = pd.to_numeric(evaluable[var], errors="coerce"), pd.to_numeric(unevaluable[var], errors="coerce")
        rows.append({
            "variable": var, "level": "continuous", "display_type": "median_iqr",
            "evaluable_n": int(eva.notna().sum()), "evaluable_value": f"{eva.median():.2f} ({eva.quantile(.25):.2f}–{eva.quantile(.75):.2f})",
            "unevaluable_n": int(uneva.notna().sum()), "unevaluable_value": f"{uneva.median():.2f} ({uneva.quantile(.25):.2f}–{uneva.quantile(.75):.2f})",
            "standardized_mean_difference": standardized_mean_difference(eva, uneva),
        })
    for var in binary:
        if var not in data:
            continue
        eva, uneva = pd.to_numeric(evaluable[var], errors="coerce").fillna(0), pd.to_numeric(unevaluable[var], errors="coerce").fillna(0)
        rows.append({
            "variable": var, "level": "yes", "display_type": "n_percent",
            "evaluable_n": int(eva.sum()), "evaluable_value": f"{int(eva.sum())} ({eva.mean()*100:.1f}%)",
            "unevaluable_n": int(uneva.sum()), "unevaluable_value": f"{int(uneva.sum())} ({uneva.mean()*100:.1f}%)",
            "standardized_mean_difference": standardized_mean_difference(eva, uneva),
        })
    for var in categorical:
        if var not in data:
            continue
        levels = sorted(set(evaluable[var].fillna("Missing").astype(str)) | set(unevaluable[var].fillna("Missing").astype(str)))
        for level in levels:
            eva = evaluable[var].fillna("Missing").astype(str).eq(level).astype(int)
            uneva = unevaluable[var].fillna("Missing").astype(str).eq(level).astype(int)
            rows.append({
                "variable": var, "level": level, "display_type": "n_percent",
                "evaluable_n": int(eva.sum()), "evaluable_value": f"{int(eva.sum())} ({eva.mean()*100:.1f}%)",
                "unevaluable_n": int(uneva.sum()), "unevaluable_value": f"{int(uneva.sum())} ({uneva.mean()*100:.1f}%)",
                "standardized_mean_difference": standardized_mean_difference(eva, uneva),
            })
    return pd.DataFrame(rows)


def choose_grouped_subject_split(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = data["creatinine_record_observable"].astype(int)
    groups = data["subject_id"].astype(str)
    overall = y.mean()
    splitter = GroupShuffleSplit(n_splits=300, test_size=.20, random_state=RANDOM_STATE)
    best = None
    for train_idx, test_idx in splitter.split(data, y, groups):
        score = abs(y.iloc[test_idx].mean() - overall) + abs(len(test_idx) / len(data) - .20)
        if best is None or score < best[0]:
            best = (score, train_idx, test_idx)
    assert best is not None
    return best[1], best[2]


def make_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ], remainder="drop")
    return Pipeline([("preprocess", prep), ("model", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE))])


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)), "creatinine_record_observable_n": int(y.sum()), "creatinine_record_observable_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)), "brier_score": float(brier_score_loss(y, p)),
    }


def observability_models(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx, test_idx = choose_grouped_subject_split(data)
    train, test = data.iloc[train_idx].copy(), data.iloc[test_idx].copy()
    model_specs = {
        "clinical_only": (NUMERIC_PREDICTORS, CATEGORICAL_PREDICTORS),
        "clinical_plus_hospital_identifier": (NUMERIC_PREDICTORS, [*CATEGORICAL_PREDICTORS, "hospitalid"]),
    }
    rows, pred_tables = [], []
    for name, (numeric, categorical) in model_specs.items():
        available_num = [x for x in numeric if x in data]
        available_cat = [x for x in categorical if x in data]
        pipe = make_pipeline(available_num, available_cat)
        pipe.fit(train[[*available_num, *available_cat]], train["creatinine_record_observable"].astype(int))
        p = pipe.predict_proba(test[[*available_num, *available_cat]])[:, 1]
        row = {"model": name, "predictor_n": len(available_num) + len(available_cat), "split": "subject-grouped held-out 20%"}
        row.update(metrics(test["creatinine_record_observable"].astype(int).to_numpy(), p))
        rows.append(row)
        pred = test[["stay_id", "subject_id", "hospitalid", "creatinine_record_observable"]].copy()
        pred["model"] = name
        pred["predicted_observability"] = p
        pred_tables.append(pred)
    return pd.DataFrame(rows), pd.concat(pred_tables, ignore_index=True)


def hospital_observability(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for hospital, group in data.groupby("hospitalid", sort=True):
        n = len(group); observed = int(group["creatinine_record_observable"].sum()); rate = observed / n
        se = np.sqrt(rate * (1 - rate) / n) if n else np.nan
        rows.append({
            "hospitalid": hospital, "n": n, "creatinine_record_observable_n": observed, "observability_rate": rate,
            "observability_ci_lower": max(0, rate - 1.96 * se), "observability_ci_upper": min(1, rate + 1.96 * se),
            "median_apache_score": float(pd.to_numeric(group["apachescore"], errors="coerce").median()),
            "hospital_death_rate": float(pd.to_numeric(group["hospital_death"], errors="coerce").mean()),
        })
    hospital = pd.DataFrame(rows)
    large = hospital.loc[hospital["n"].ge(100)].copy()
    summary = pd.DataFrame([{
        "hospital_n": int(len(hospital)), "hospital_n_ge_100": int(hospital["n"].ge(100).sum()),
        "observability_median": float(hospital["observability_rate"].median()),
        "observability_iqr_lower": float(hospital["observability_rate"].quantile(.25)),
        "observability_iqr_upper": float(hospital["observability_rate"].quantile(.75)),
        "observability_range_lower": float(hospital["observability_rate"].min()),
        "observability_range_upper": float(hospital["observability_rate"].max()),
        "large_hospital_observability_median": float(large["observability_rate"].median()),
        "large_hospital_observability_iqr_lower": float(large["observability_rate"].quantile(.25)),
        "large_hospital_observability_iqr_upper": float(large["observability_rate"].quantile(.75)),
        "large_hospital_observability_range_lower": float(large["observability_rate"].min()),
        "large_hospital_observability_range_upper": float(large["observability_rate"].max()),
    }])
    return hospital, summary


def data_quality_audit(data: pd.DataFrame, apache: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"check": "row_count", "value": len(data), "note": "Strict eICU surgical first-ICU cohort"},
        {"check": "duplicate_stay_id", "value": int(data.duplicated("stay_id").sum()), "note": "Must be zero"},
        {"check": "unique_hospitals", "value": int(data["hospitalid"].nunique()), "note": "Institutional measurement-practice source"},
        {"check": "analysis_evaluability_rate", "value": float(data["outcome_evaluable"].mean()), "note": "AKI analysis-eligibility outcome"},
        {"check": "creatinine_record_observability_rate", "value": float(data["creatinine_record_observable"].mean()), "note": "Measurement-selection outcome: baseline plus post-index SCr record"},
        {"check": "apache_iva_join_coverage", "value": float(data["apachescore"].notna().mean()), "note": "Coverage after one-to-one APACHE IVa join"},
        {"check": "hospital_mortality_status_coverage", "value": float(data["hospital_death"].notna().mean()), "note": "Hospital death field coverage"},
        {"check": "apache_iva_duplicate_stay_id", "value": int(apache.duplicated("stay_id").sum()), "note": "Must be zero after IVa selection"},
    ]
    return pd.DataFrame(rows)


def make_figures(comparison: pd.DataFrame, obs_predictions: pd.DataFrame, hospitals: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    # High-signal balance plot: variables with the largest group difference.
    smd = comparison.copy()
    smd = smd.loc[smd["variable"].isin(["anchor_age", "charlson_score", "acutephysiologyscore", "apachescore", "predictedhospitalmortality", "icu_death", "hospital_death", "ckd", "chf", "dm", "copd", "cancer"])].dropna(subset=["standardized_mean_difference"])
    clinical_labels = {
        "anchor_age": "Age", "charlson_score": "Charlson comorbidity score",
        "acutephysiologyscore": "Acute physiology score", "apachescore": "APACHE score",
        "predictedhospitalmortality": "Predicted hospital mortality", "icu_death": "ICU death",
        "hospital_death": "Hospital death", "ckd": "Chronic kidney disease",
        "chf": "Congestive heart failure", "dm": "Diabetes mellitus", "copd": "COPD",
        "cancer": "Cancer",
    }
    smd["label"] = smd.apply(lambda r: clinical_labels.get(r["variable"], r["variable"]) if r["level"] in {"continuous", "yes"} else f"{clinical_labels.get(r['variable'], r['variable'])}: {r['level']}", axis=1)
    smd = smd.sort_values("standardized_mean_difference")
    fig, ax = plt.subplots(figsize=(7.6, max(4.5, len(smd) * .34)), dpi=180)
    y = np.arange(len(smd))
    colors = np.where(smd["standardized_mean_difference"] >= 0, "#4c78a8", "#e17c05")
    ax.hlines(y, 0, smd["standardized_mean_difference"], color=colors, linewidth=2)
    ax.scatter(smd["standardized_mean_difference"], y, color=colors, s=32)
    ax.axvline(0, color="#30343b", linewidth=1)
    ax.axvline(.1, color="#687080", linestyle="--", linewidth=.8)
    ax.axvline(-.1, color="#687080", linestyle="--", linewidth=.8)
    ax.set_yticks(y, smd["label"])
    ax.set_xlabel("Standardized mean difference: evaluable minus unevaluable")
    ax.set_title("Patient differences by AKI-outcome observability")
    fig.text(.5, .01, "Positive values indicate higher prevalence/value among outcome-evaluable stays; dashed lines denote |SMD| = 0.10", ha="center", fontsize=8.5, color="#5f6675")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUTPUT_DIR / "figure_v18_observability_group_differences.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=180)
    for model, group in obs_predictions.groupby("model"):
        y = group["creatinine_record_observable"].astype(int)
        p = group["predicted_observability"]
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        label = "Clinical only" if model == "clinical_only" else "Clinical + hospital identifier"
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUROC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#687080", linewidth=1)
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("Predictability of creatinine-record observability")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.text(.5, .01, "Held-out patients; outcome is creatinine-record observability, not AKI", ha="center", fontsize=8.5, color="#5f6675")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUTPUT_DIR / "figure_v18_observability_prediction_roc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    plot = hospitals.loc[hospitals["n"].ge(100)].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    size = np.clip(plot["n"] / max(plot["n"].max(), 1) * 160, 24, 160)
    ax.scatter(plot["median_apache_score"], plot["observability_rate"] * 100, s=size, color="#4c78a8", alpha=.78, edgecolor="white", linewidth=.5)
    ax.set_xlabel("Hospital median APACHE score")
    ax.set_ylabel("Creatinine-record observability (%)")
    ax.set_title("Hospital variation in creatinine-record observability")
    fig.text(.5, .01, "Hospitals with ≥100 strict surgical ICU stays; point area proportional to sample size", ha="center", fontsize=8.5, color="#5f6675")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUTPUT_DIR / "figure_v18_hospital_observability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_readme(data: pd.DataFrame, comparison: pd.DataFrame, performance: pd.DataFrame, hospital_summary: pd.DataFrame) -> None:
    eva = data.loc[data["outcome_evaluable"].eq(1)]
    uneva = data.loc[data["outcome_evaluable"].eq(0)]
    lines = ["# v18 eICU outcome-observability and measurement-selection audit", "", "## Question", ""]
    lines.append("Why were some strict postoperative eICU stays evaluable for the SCr-based incident AKI outcome whereas others were not? This is a measurement-process audit; unevaluable stays are not assumed to be free of AKI.")
    lines.extend(["", "## Cohort and outcome-observability definition", ""])
    creatinine_observed = data.loc[data["creatinine_record_observable"].eq(1)]
    lines.append(f"The audit included {len(data):,} strict eICU surgical first ICU stays. Analysis-evaluable status required a baseline SCr, absence of pre-index AKI, and at least one post-index SCr within 7 days. {len(eva):,} ({len(eva)/len(data)*100:.1f}%) were analysis-evaluable and {len(uneva):,} ({len(uneva)/len(data)*100:.1f}%) were not. The separate measurement-selection outcome—baseline plus post-index SCr record regardless of pre-index AKI—was observed in {len(creatinine_observed):,} ({len(creatinine_observed)/len(data)*100:.1f}%).")
    lines.extend(["", "## Measurement-selection prediction", ""])
    for r in performance.itertuples():
        label = "clinical variables only" if r.model == "clinical_only" else "clinical variables plus hospital identifier"
        lines.append(f"- {label}: held-out AUROC {r.auroc:.3f}, AUPRC {r.auprc:.3f}, Brier {r.brier_score:.3f} for predicting creatinine-record observability.")
    lines.extend(["", "## Hospital-level variation", ""])
    r = hospital_summary.iloc[0]
    lines.append(f"Across {int(r.hospital_n)} hospitals, median creatinine-record observability was {r.observability_median*100:.1f}% (IQR {r.observability_iqr_lower*100:.1f}%–{r.observability_iqr_upper*100:.1f}%). Among the {int(r.hospital_n_ge_100)} hospitals with ≥100 strict surgical ICU stays, median observability was {r.large_hospital_observability_median*100:.1f}% (IQR {r.large_hospital_observability_iqr_lower*100:.1f}%–{r.large_hospital_observability_iqr_upper*100:.1f}%; range {r.large_hospital_observability_range_lower*100:.1f}%–{r.large_hospital_observability_range_upper*100:.1f}%).")
    lines.extend(["", "## Clinical comparison", ""])
    lines.append(f"Compared with unevaluable stays, evaluable stays had a higher median APACHE score ({pd.to_numeric(eva['apachescore'], errors='coerce').median():.0f} vs {pd.to_numeric(uneva['apachescore'], errors='coerce').median():.0f}) and higher median predicted hospital mortality ({pd.to_numeric(eva['predictedhospitalmortality'], errors='coerce').median()*100:.1f}% vs {pd.to_numeric(uneva['predictedhospitalmortality'], errors='coerce').median()*100:.1f}%). Observed hospital mortality was {pd.to_numeric(eva['hospital_death'], errors='coerce').mean()*100:.1f}% vs {pd.to_numeric(uneva['hospital_death'], errors='coerce').mean()*100:.1f}%, respectively. Full surgery-system, ICU-type, comorbidity, mortality, and severity comparisons are in `audit_v18_evaluable_vs_unevaluable_comparison.csv`.")
    lines.extend(["", "## Interpretation", ""])
    lines.append("If clinical characteristics or hospital identity predict creatinine-record observability, complete-case external validation is susceptible to measurement-selection bias. This audit quantifies the limitation but does not correct it, because the AKI outcome is unobserved when no adequate SCr record exists. Results should be reported as generalizing most directly to postoperative ICU patients with documented baseline and follow-up SCr monitoring.")
    lines.extend(["", "## Key caveats", "", "- APACHE IVa measures are used as admission severity proxies; missingness remains explicit and is imputed only inside the observability prediction model.", "- Hospital discharge mortality is used for descriptive group comparison only, never as a pre-index predictor.", "- The hospital-identifier observability model measures institutional documentation patterns within eICU and is not intended to transport to a new hospital."])
    (OUTPUT_DIR / "audit_v18_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(V16_DIR / "cohort_v16_eicu_external_validation.csv.gz", low_memory=False).copy()
    if cohort.duplicated("stay_id").any():
        raise ValueError("v16 external cohort is not unique at ICU-stay grain")
    cohort["outcome_evaluable"] = cohort["incident_aki_evaluable"].fillna(False).astype(int)
    cohort["ineligibility_reason"] = cohort["ineligibility_reason"].fillna("baseline_scr_missing")
    cohort["creatinine_record_observable"] = (
        cohort["baseline_scr_source"].fillna("missing").ne("missing")
        & pd.to_numeric(cohort["post_index_scr_n_7d"], errors="coerce").fillna(0).gt(0)
    ).astype(int)
    patient, apache = read_enrichment(set(cohort["stay_id"].astype(int)))
    data = cohort.merge(patient, on="stay_id", how="left", validate="one_to_one").merge(apache, on="stay_id", how="left", validate="one_to_one")
    comparison = group_comparison(data)
    performance, predicted = observability_models(data)
    hospitals, hospital_summary = hospital_observability(data)
    quality = data_quality_audit(data, apache)
    comparison.to_csv(OUTPUT_DIR / "audit_v18_evaluable_vs_unevaluable_comparison.csv", index=False)
    performance.to_csv(OUTPUT_DIR / "model_v18_outcome_observability_performance.csv", index=False)
    predicted.to_csv(OUTPUT_DIR / "model_v18_outcome_observability_test_predictions.csv", index=False)
    hospitals.to_csv(OUTPUT_DIR / "analysis_v18_hospital_observability.csv", index=False)
    hospital_summary.to_csv(OUTPUT_DIR / "analysis_v18_hospital_observability_summary.csv", index=False)
    quality.to_csv(OUTPUT_DIR / "audit_v18_data_quality.csv", index=False)
    make_figures(comparison, predicted, hospitals)
    write_readme(data, comparison, performance, hospital_summary)
    print(f"Wrote v18 observability outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
