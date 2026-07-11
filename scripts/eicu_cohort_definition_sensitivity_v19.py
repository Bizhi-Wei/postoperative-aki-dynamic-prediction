"""v19 eICU external-cohort definition sensitivity analysis.

The MIMIC-trained portable models remain frozen.  This script compares external
performance across four transparent eICU cohort definitions:

1. v16 strict operative-system cohort (reference);
2. strict cohort with ICU unit-visit number 1 only;
3. strict cohort restricted to surgical/cardiac/neuro ICU types;
4. broad all-Operative-Organ-Systems cohort.

The broad cohort is derived independently from eICU raw tables; no MIMIC raw
data are read.  All variants use the same SCr-only KDIGO outcome algorithm and
landmark risk-set rules as v16.
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from develop_models_v5 import OUTCOME, load_data  # noqa: E402
from recalibration_and_measurement_intensity_v13 import identify_types_for_columns  # noqa: E402
from external_validation_eicu_v16 import (  # noqa: E402
    CHUNK_SIZE, EICU_ROOT, FOLLOWUP_MINUTES, LAB_FILE, PATIENT_FILE, ADMISSION_DX_FILE,
    DIAGNOSIS_FILE, PRE24_MINUTES, PRE_LOOKBACK_MINUTES, VITAL_FILE,
    derive_aki, extract_eicu_diagnoses, lab_window_features, make_model,
    portable_predictors, preindex_lab_features, vital_window_features,
    eicu_landmark_dataset,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = SCRIPT_DIR.parent
V16_DIR = PROJECT_ROOT / "outputs" / "modeling_v16_eicu_external_validation"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v19_eicu_cohort_definition_sensitivity"
CACHE_DIR = OUTPUT_DIR / "cache"
OPERATIVE_PREFIX = "admission diagnosis|Operative Organ Systems|Organ System|"
STRICT_SYSTEMS = {"Cardiovascular", "Gastrointestinal", "Neurologic", "Respiratory", "Musculoskeletal/Skin"}
SURGICAL_UNIT_TYPES = {"SICU", "CTICU", "CCU-CTICU", "CSICU", "Neuro ICU", "Cardiac ICU"}


def age_to_numeric(value: object) -> float:
    text = str(value).strip()
    if text in {">89", "> 89"}:
        return 90.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def broad_operative_index() -> pd.DataFrame:
    patient = pd.read_csv(
        PATIENT_FILE,
        usecols=["patientunitstayid", "patienthealthsystemstayid", "gender", "age", "hospitalid", "unittype", "unitvisitnumber", "uniquepid"],
        low_memory=False,
    )
    patient["patientunitstayid"] = pd.to_numeric(patient["patientunitstayid"], errors="coerce").astype("Int64")
    patient["patienthealthsystemstayid"] = pd.to_numeric(patient["patienthealthsystemstayid"], errors="coerce").astype("Int64")
    patient["unitvisitnumber"] = pd.to_numeric(patient["unitvisitnumber"], errors="coerce")
    patient["anchor_age"] = patient["age"].map(age_to_numeric)
    patient = patient.loc[patient["anchor_age"].ge(18)].copy()
    parts = []
    for i, chunk in enumerate(pd.read_csv(ADMISSION_DX_FILE, usecols=["patientunitstayid", "admitdxpath"], chunksize=CHUNK_SIZE, low_memory=False), 1):
        path = chunk["admitdxpath"].fillna("")
        sub = chunk.loc[path.str.startswith(OPERATIVE_PREFIX)].copy()
        if not sub.empty:
            sub["operative_system"] = sub["admitdxpath"].str.replace(OPERATIVE_PREFIX, "", regex=False)
            parts.append(sub[["patientunitstayid", "operative_system"]])
        if i % 5 == 0:
            print(f"Scanned {i * CHUNK_SIZE:,} eICU admission-diagnosis rows for broad cohort...", flush=True)
    operative = pd.concat(parts, ignore_index=True).drop_duplicates()
    operative["patientunitstayid"] = pd.to_numeric(operative["patientunitstayid"], errors="coerce").astype("Int64")
    index = patient.merge(operative, on="patientunitstayid", how="inner", validate="one_to_many")
    index = index.sort_values(["patienthealthsystemstayid", "unitvisitnumber", "patientunitstayid"])
    # Broad sensitivity: first ICU stay carrying any operative-organ-system label.
    index = index.drop_duplicates("patienthealthsystemstayid", keep="first").copy()
    index = index.rename(columns={"patientunitstayid": "stay_id", "uniquepid": "subject_id"})
    index["stay_id"] = index["stay_id"].astype(int)
    index["subject_id"] = index["subject_id"].astype("string")
    index["gender"] = index["gender"].map({"Male": "M", "Female": "F"}).fillna("Unknown")
    index["cardiac_surgery"] = index["operative_system"].eq("Cardiovascular").astype(int)
    index["non_cardiac_surgery"] = (1 - index["cardiac_surgery"]).astype(int)
    index["general_gi_hepatobiliary_surgery"] = index["operative_system"].eq("Gastrointestinal").astype(int)
    index["orthopedic_major_surgery"] = index["operative_system"].eq("Musculoskeletal/Skin").astype(int)
    index["neurosurgery"] = index["operative_system"].eq("Neurologic").astype(int)
    index["thoracic_respiratory_surgery"] = index["operative_system"].eq("Respiratory").astype(int)
    return index.reset_index(drop=True)


def broad_labs(index: pd.DataFrame) -> pd.DataFrame:
    cache = CACHE_DIR / "broad_operative_selected_labs.csv.gz"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    ids = set(index["stay_id"].astype(int))
    target_names = {"creatinine", "BUN", "Hgb", "lactate", "WBC x 1000", "platelets x 1000", "potassium", "sodium"}
    retained = []
    for i, chunk in enumerate(pd.read_csv(LAB_FILE, usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"], chunksize=CHUNK_SIZE, low_memory=False), 1):
        chunk["patientunitstayid"] = pd.to_numeric(chunk["patientunitstayid"], errors="coerce")
        chunk["labresultoffset"] = pd.to_numeric(chunk["labresultoffset"], errors="coerce")
        sub = chunk.loc[chunk["patientunitstayid"].isin(ids) & chunk["labname"].isin(target_names) & chunk["labresultoffset"].between(-PRE_LOOKBACK_MINUTES, FOLLOWUP_MINUTES)].copy()
        if not sub.empty:
            retained.append(sub)
        if i % 10 == 0:
            print(f"Scanned {i * CHUNK_SIZE:,} eICU laboratory rows for broad cohort...", flush=True)
    labs = pd.concat(retained, ignore_index=True) if retained else pd.DataFrame(columns=["patientunitstayid", "labresultoffset", "labname", "labresult"])
    labs = labs.rename(columns={"patientunitstayid": "stay_id", "labresultoffset": "offset_min", "labresult": "value"})
    labs["stay_id"] = pd.to_numeric(labs["stay_id"], errors="coerce").astype(int)
    labs["offset_min"] = pd.to_numeric(labs["offset_min"], errors="coerce")
    labs["value"] = pd.to_numeric(labs["value"], errors="coerce")
    labs = labs.dropna(subset=["offset_min", "value"]).sort_values(["stay_id", "labname", "offset_min"])
    labs.to_csv(cache, index=False, compression="gzip")
    return labs


def broad_vitals(index: pd.DataFrame) -> pd.DataFrame:
    cache = CACHE_DIR / "broad_operative_selected_vitals.csv.gz"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    ids = set(index["stay_id"].astype(int))
    retained = []
    cols = ["patientunitstayid", "observationoffset", "sao2", "heartrate", "systemicsystolic", "systemicmean"]
    for i, chunk in enumerate(pd.read_csv(VITAL_FILE, usecols=cols, chunksize=CHUNK_SIZE, low_memory=False), 1):
        chunk["patientunitstayid"] = pd.to_numeric(chunk["patientunitstayid"], errors="coerce")
        chunk["observationoffset"] = pd.to_numeric(chunk["observationoffset"], errors="coerce")
        sub = chunk.loc[chunk["patientunitstayid"].isin(ids) & chunk["observationoffset"].gt(0) & chunk["observationoffset"].le(PRE24_MINUTES)].copy()
        if not sub.empty:
            retained.append(sub)
        if i % 10 == 0:
            print(f"Scanned {i * CHUNK_SIZE:,} eICU periodic-vital rows for broad cohort...", flush=True)
    vitals = pd.concat(retained, ignore_index=True) if retained else pd.DataFrame(columns=cols)
    vitals = vitals.rename(columns={"patientunitstayid": "stay_id", "observationoffset": "offset_min"})
    vitals["stay_id"] = pd.to_numeric(vitals["stay_id"], errors="coerce").astype("Int64")
    vitals["offset_min"] = pd.to_numeric(vitals["offset_min"], errors="coerce")
    for col in ["sao2", "heartrate", "systemicsystolic", "systemicmean"]:
        vitals[col] = pd.to_numeric(vitals[col], errors="coerce")
    vitals = vitals.dropna(subset=["stay_id", "offset_min"])
    vitals.to_csv(cache, index=False, compression="gzip")
    return vitals


def broad_cohort() -> pd.DataFrame:
    cache = CACHE_DIR / "cohort_broad_operative_harmonized.csv.gz"
    if cache.exists():
        cohort = pd.read_csv(cache, low_memory=False)
        cohort["incident_aki_evaluable"] = cohort["incident_aki_evaluable"].fillna(False)
        cohort["ineligibility_reason"] = cohort["ineligibility_reason"].fillna("baseline_scr_missing")
        cohort["baseline_scr_source"] = cohort["baseline_scr_source"].fillna("missing")
        return cohort
    index = broad_operative_index()
    print(f"Broad eICU operative index cohort: {len(index):,} first operative ICU stays.", flush=True)
    dx = extract_eicu_diagnoses(index)
    labs = broad_labs(index)
    print(f"Retained {len(labs):,} broad-cohort laboratory rows.", flush=True)
    vitals = broad_vitals(index)
    print(f"Retained {len(vitals):,} broad-cohort periodic-vital rows.", flush=True)
    aki = derive_aki(index, labs)
    cohort = index.merge(dx, on="stay_id", how="left", validate="one_to_one").merge(aki, on="stay_id", how="left", validate="one_to_one")
    cohort["incident_aki_evaluable"] = cohort["incident_aki_evaluable"].fillna(False)
    cohort["ineligibility_reason"] = cohort["ineligibility_reason"].fillna("baseline_scr_missing")
    cohort["baseline_scr_source"] = cohort["baseline_scr_source"].fillna("missing")
    cohort = cohort.set_index("stay_id", drop=False)
    frames = [
        preindex_lab_features(labs), lab_window_features(labs, 0, 6 * 60, "0_6h"),
        lab_window_features(labs, 0, 24 * 60, "0_24h"), vital_window_features(vitals, 6 * 60, "0_6h"),
        vital_window_features(vitals, 24 * 60, "0_24h"),
    ]
    for frame in frames:
        cohort = cohort.join(frame, how="left")
    cohort = cohort.reset_index(drop=True)
    cohort.to_csv(cache, index=False, compression="gzip")
    return cohort


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from recalibration_and_measurement_intensity_v13 import calibration_intercept_slope
    y = np.asarray(y, dtype=int); p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    intercept, slope = calibration_intercept_slope(y, p)
    return {"n": int(len(y)), "event_n": int(y.sum()), "event_rate": float(y.mean()), "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)), "brier_score": float(brier_score_loss(y, p)), "calibration_intercept": intercept, "calibration_slope": slope, "mean_predicted_risk": float(p.mean())}


def bootstrap_auroc(y: np.ndarray, p: np.ndarray, draws: int = 300, seed: int = 20250709) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def frozen_broad_predictions(cohort: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for landmark in (0, 6, 24):
        predictors = portable_predictors(landmark)
        dev = load_data(landmark)
        ext = eicu_landmark_dataset(cohort, landmark, predictors)
        continuous, binary, categorical = identify_types_for_columns(dev[predictors], predictors)
        model = make_model(landmark, continuous, binary, categorical)
        model.fit(dev[predictors], dev[OUTCOME].astype(int).to_numpy())
        p = model.predict_proba(ext[predictors])[:, 1]
        out = ext[["stay_id", "subject_id", "hospitalid", "operative_system", "unittype", "unitvisitnumber", OUTCOME]].copy()
        out["landmark_hours"] = landmark
        out["predicted_risk_portable_model"] = p
        tables.append(out)
    return pd.concat(tables, ignore_index=True)


def evaluate_variant(predictions: pd.DataFrame, variant: str) -> tuple[list[dict[str, object]], pd.DataFrame]:
    rows, tables = [], []
    for landmark in (0, 6, 24):
        data = predictions.loc[predictions["landmark_hours"].eq(landmark)].copy()
        y = data[OUTCOME].astype(int).to_numpy(); p = data["predicted_risk_portable_model"].to_numpy()
        row = {"cohort_variant": variant, "landmark_hours": landmark}
        row.update(metrics(y, p))
        low, high = bootstrap_auroc(y, p, seed=20250709 + landmark + len(variant))
        row.update({"auroc_ci_lower": low, "auroc_ci_upper": high})
        rows.append(row)
        data["cohort_variant"] = variant
        tables.append(data)
    return rows, pd.concat(tables, ignore_index=True)


def make_figures(performance: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    variants = ["strict_v16_reference", "strict_unitvisit1_only", "strict_surgical_icu_types", "broad_all_operative_systems"]
    labels = {"strict_v16_reference": "Strict reference", "strict_unitvisit1_only": "Strict + first ICU visit", "strict_surgical_icu_types": "Strict + surgical ICU types", "broad_all_operative_systems": "Broad operative systems"}
    colors = {"strict_v16_reference": "#4c78a8", "strict_unitvisit1_only": "#59a14f", "strict_surgical_icu_types": "#e17c05", "broad_all_operative_systems": "#af7aa1"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=180, sharey=True)
    for ax, landmark in zip(axes, (0, 6, 24)):
        d = performance.loc[performance["landmark_hours"].eq(landmark)].set_index("cohort_variant").loc[variants].reset_index()
        y = np.arange(len(d))
        ax.errorbar(d["auroc"], y, xerr=[d["auroc"] - d["auroc_ci_lower"], d["auroc_ci_upper"] - d["auroc"]], fmt="none", color="#30343b", capsize=3)
        ax.scatter(d["auroc"], y, s=55, c=[colors[x] for x in d["cohort_variant"]])
        ax.set_yticks(y, [labels[x] for x in d["cohort_variant"]])
        ax.set_xlim(.55, .78)
        ax.set_title(f"{landmark} h")
        ax.set_xlabel("AUROC")
    fig.suptitle("External discrimination across eICU cohort definitions", y=1.03)
    fig.text(.5, .965, "Frozen MIMIC-trained portable models; 95% bootstrap confidence intervals", ha="center", fontsize=9, color="#5f6675")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUTPUT_DIR / "figure_v19_cohort_definition_auroc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=180)
    for variant, group in performance.groupby("cohort_variant", sort=False):
        group = group.sort_values("landmark_hours")
        ax.plot(group["landmark_hours"], group["calibration_slope"], marker="o", linewidth=1.8, color=colors[variant], label=labels[variant])
    ax.axhline(1, linestyle="--", color="#687080", linewidth=1)
    ax.set_xticks([0, 6, 24], ["0 h", "6 h", "24 h"])
    ax.set_ylabel("Calibration slope")
    ax.set_title("External calibration across cohort definitions")
    ax.legend(frameon=False, fontsize=8)
    fig.text(.5, .01, "Frozen portable models; calibration slope of 1 indicates ideal spread", ha="center", fontsize=8.5, color="#5f6675")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUTPUT_DIR / "figure_v19_cohort_definition_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_readme(performance: pd.DataFrame, definition: pd.DataFrame) -> None:
    lines = ["# v19 eICU external cohort-definition sensitivity analysis", "", "## Cohort definitions", ""]
    for row in definition.itertuples():
        lines.append(f"- **{row.cohort_variant}**: {row.definition}")
    lines.extend(["", "## External performance", ""])
    for row in performance.sort_values(["cohort_variant", "landmark_hours"]).itertuples():
        lines.append(f"- {row.cohort_variant}, {int(row.landmark_hours)} h: n={int(row.n):,}, events={int(row.event_n):,} ({row.event_rate*100:.1f}%), AUROC {row.auroc:.3f} (95% CI {row.auroc_ci_lower:.3f}–{row.auroc_ci_upper:.3f}), AUPRC {row.auprc:.3f}, Brier {row.brier_score:.3f}, slope {row.calibration_slope:.2f}.")
    lines.extend(["", "## Interpretation", "", "The primary eICU result remains the v16 strict reference cohort. This sensitivity analysis assesses whether external performance depends materially on reasonable operational choices for postoperative cohort definition. Broad operative systems include categories not mapped to the primary MIMIC surgical taxonomy and should be interpreted as a transportability sensitivity, not a replacement primary cohort."])
    (OUTPUT_DIR / "audit_v19_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); CACHE_DIR.mkdir(parents=True, exist_ok=True)
    strict_cohort = pd.read_csv(V16_DIR / "cohort_v16_eicu_external_validation.csv.gz", low_memory=False)
    strict_pred = pd.read_csv(V16_DIR / "model_v16_portable_external_predictions.csv", low_memory=False)
    strict_pred = strict_pred.merge(strict_cohort[["stay_id", "unitvisitnumber", "unittype"]], on="stay_id", how="left", validate="many_to_one")
    variants = {
        "strict_v16_reference": strict_pred,
        "strict_unitvisit1_only": strict_pred.loc[pd.to_numeric(strict_pred["unitvisitnumber"], errors="coerce").eq(1)].copy(),
        "strict_surgical_icu_types": strict_pred.loc[strict_pred["unittype"].isin(SURGICAL_UNIT_TYPES)].copy(),
    }
    broad = broad_cohort()
    broad_pred_cache = CACHE_DIR / "broad_operative_frozen_predictions.csv.gz"
    if broad_pred_cache.exists():
        broad_pred = pd.read_csv(broad_pred_cache, low_memory=False)
    else:
        broad_pred = frozen_broad_predictions(broad)
        broad_pred.to_csv(broad_pred_cache, index=False, compression="gzip")
    variants["broad_all_operative_systems"] = broad_pred
    rows, tables = [], []
    for name, pred in variants.items():
        r, t = evaluate_variant(pred, name)
        rows.extend(r); tables.append(t)
    performance = pd.DataFrame(rows)
    predictions = pd.concat(tables, ignore_index=True)
    definition = pd.DataFrame([
        {"cohort_variant": "strict_v16_reference", "definition": "Five prespecified operative systems (cardiovascular, gastrointestinal, neurologic, respiratory, musculoskeletal/skin); first ICU stay carrying a strict surgical-system label per hospital stay."},
        {"cohort_variant": "strict_unitvisit1_only", "definition": "Strict v16 cohort restricted to ICU unit-visit number 1."},
        {"cohort_variant": "strict_surgical_icu_types", "definition": "Strict v16 cohort restricted to SICU, CTICU, CCU-CTICU, CSICU, Neuro ICU, or Cardiac ICU."},
        {"cohort_variant": "broad_all_operative_systems", "definition": "First ICU stay carrying any eICU Operative Organ Systems label, including genitourinary, trauma, transplant, metabolic/endocrine, and hematology systems."},
    ])
    audit = pd.DataFrame([
        {"check": "strict_prediction_duplicate_stay_landmark", "value": int(strict_pred.duplicated(["stay_id", "landmark_hours"]).sum()), "note": "Must be zero"},
        {"check": "broad_prediction_duplicate_stay_landmark", "value": int(broad_pred.duplicated(["stay_id", "landmark_hours"]).sum()), "note": "Must be zero"},
        {"check": "broad_cohort_duplicate_stay", "value": int(broad.duplicated("stay_id").sum()), "note": "Must be zero"},
        {"check": "broad_operating_system_n", "value": int(broad["operative_system"].nunique()), "note": "Distinct eICU Operative Organ Systems categories"},
    ])
    performance.to_csv(OUTPUT_DIR / "analysis_v19_cohort_definition_sensitivity_performance.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "analysis_v19_cohort_definition_sensitivity_predictions.csv.gz", index=False, compression="gzip")
    definition.to_csv(OUTPUT_DIR / "audit_v19_cohort_definitions.csv", index=False)
    audit.to_csv(OUTPUT_DIR / "audit_v19_data_quality.csv", index=False)
    broad.to_csv(OUTPUT_DIR / "cohort_v19_eicu_broad_operative.csv.gz", index=False, compression="gzip")
    make_figures(performance)
    write_readme(performance, definition)
    print(f"Wrote v19 cohort-definition sensitivity outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
