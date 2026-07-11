"""v16 feature-harmonized external validation in eICU-CRD v2.0.

The primary MIMIC-IV models are not altered.  eICU differs materially in its
surgical coding, diagnosis coding, and data schema; therefore, this script
validates *portable, feature-harmonized versions* of the selected models.
Portable models are fit on all MIMIC development data using only variables
with an explicit eICU mapping, then frozen and evaluated once in eICU.

This is a multi-centre external validation, not a retraining exercise on eICU.
No MIMIC raw tables are read: the MIMIC modelling-ready datasets are used.
"""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import xgboost as xgb

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from develop_models_v5 import OUTCOME, RANDOM_STATE, choose_grouped_split, load_data  # noqa: E402
from recalibration_and_measurement_intensity_v13 import (  # noqa: E402
    calibration_intercept_slope,
    identify_types_for_columns,
    make_preprocessor_custom,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = SCRIPT_DIR.parent
EICU_ROOT = Path(os.environ.get("EICU_ROOT", str(PROJECT_ROOT.parents[1] / "eicu-collaborative-research-database-2.0")))
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v16_eicu_external_validation"
CACHE_DIR = OUTPUT_DIR / "cache"

PATIENT_FILE = EICU_ROOT / "patient.csv" / "patient.csv"
ADMISSION_DX_FILE = EICU_ROOT / "admissionDx.csv" / "admissionDx.csv"
DIAGNOSIS_FILE = EICU_ROOT / "diagnosis.csv" / "diagnosis.csv"
LAB_FILE = EICU_ROOT / "lab.csv" / "lab.csv"
VITAL_FILE = EICU_ROOT / "vitalPeriodic.csv"

CHUNK_SIZE = 1_000_000
FOLLOWUP_MINUTES = 7 * 24 * 60
PRE_LOOKBACK_MINUTES = 7 * 24 * 60
PRE24_MINUTES = 24 * 60

# These variables exist in both databases with a defensible semantic mapping.
PORTABLE_0H = [
    "gender", "anchor_age", "chf", "hypertension", "dm", "ckd", "copd",
    "liver", "cancer", "pvd", "stroke", "mi", "obesity", "anemia",
    "charlson_score", "cardiac_surgery", "non_cardiac_surgery",
    "general_gi_hepatobiliary_surgery", "orthopedic_major_surgery",
    "neurosurgery", "thoracic_respiratory_surgery", "baseline_scr_at_landmark",
    "lab_pre24h_bun_last", "lab_pre24h_creatinine_last",
    "lab_pre24h_hemoglobin_last", "lab_pre24h_lactate_last",
    "lab_pre24h_wbc_last", "lab_pre24h_platelet_last",
    "lab_pre24h_potassium_last", "lab_pre24h_sodium_last",
]

LAB_BASES = ["bun", "creatinine", "hemoglobin", "lactate", "wbc", "platelet", "potassium", "sodium"]
VITAL_BASES = ["map", "heart_rate", "sbp", "spo2"]


def norm_bool(x: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(x):
        return x.fillna(False).astype(int)
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce").fillna(0).astype(int)
    return x.astype("string").str.lower().isin(["1", "true", "yes"]).astype(int)


def age_to_numeric(value: object) -> float:
    text = str(value).strip()
    if text in {"> 89", ">89"}:
        return 90.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def empty_feature_frame(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(index=index)


def read_eicu_surgical_index() -> pd.DataFrame:
    """Create one strict operative ICU stay per eICU hospital stay."""
    patient = pd.read_csv(
        PATIENT_FILE,
        usecols=[
            "patientunitstayid", "patienthealthsystemstayid", "gender", "age",
            "ethnicity", "hospitalid", "unittype",
            "unitvisitnumber", "unitstaytype", "uniquepid",
        ],
        low_memory=False,
    )
    patient["patientunitstayid"] = pd.to_numeric(patient["patientunitstayid"], errors="coerce").astype("Int64")
    patient["patienthealthsystemstayid"] = pd.to_numeric(patient["patienthealthsystemstayid"], errors="coerce").astype("Int64")
    patient["unitvisitnumber"] = pd.to_numeric(patient["unitvisitnumber"], errors="coerce")
    patient["anchor_age"] = patient["age"].map(age_to_numeric)
    patient = patient.loc[patient["anchor_age"].ge(18)].copy()

    parts = []
    target_path = "admission diagnosis|Operative Organ Systems|Organ System|"
    for i, chunk in enumerate(
        pd.read_csv(ADMISSION_DX_FILE, usecols=["patientunitstayid", "admitdxpath"], chunksize=CHUNK_SIZE, low_memory=False), 1
    ):
        path = chunk["admitdxpath"].fillna("")
        keep = path.str.startswith(target_path)
        if keep.any():
            sub = chunk.loc[keep].copy()
            sub["operative_system"] = sub["admitdxpath"].str.replace(target_path, "", regex=False)
            parts.append(sub[["patientunitstayid", "operative_system"]])
        if i % 5 == 0:
            print(f"Scanned {i * CHUNK_SIZE:,} eICU admission-diagnosis rows...", flush=True)
    operative = pd.concat(parts, ignore_index=True).drop_duplicates()
    strict_systems = {"Cardiovascular", "Gastrointestinal", "Neurologic", "Respiratory", "Musculoskeletal/Skin"}
    operative = operative.loc[operative["operative_system"].isin(strict_systems)].copy()
    operative["patientunitstayid"] = pd.to_numeric(operative["patientunitstayid"], errors="coerce").astype("Int64")

    index = patient.merge(operative, on="patientunitstayid", how="inner", validate="one_to_many")
    index = index.sort_values(["patienthealthsystemstayid", "unitvisitnumber", "patientunitstayid"])
    # Unit visit 1 / lowest available ICU visit number is the closest eICU analogue
    # of first ICU stay per hospital admission. Surgical status is adjudicated
    # before this deduplication to avoid retaining a preceding nonoperative stay.
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


def extract_eicu_diagnoses(index: pd.DataFrame) -> pd.DataFrame:
    """Map eICU diagnosis text to documented, broad comorbidity indicators."""
    ids = set(index["stay_id"].astype(int))
    retained = []
    for i, chunk in enumerate(
        pd.read_csv(DIAGNOSIS_FILE, usecols=["patientunitstayid", "diagnosisstring", "icd9code"], chunksize=CHUNK_SIZE, low_memory=False), 1
    ):
        chunk["patientunitstayid"] = pd.to_numeric(chunk["patientunitstayid"], errors="coerce")
        sub = chunk.loc[chunk["patientunitstayid"].isin(ids)].copy()
        if not sub.empty:
            retained.append(sub)
        if i % 5 == 0:
            print(f"Scanned {i * CHUNK_SIZE:,} eICU diagnosis rows...", flush=True)
    dx = pd.concat(retained, ignore_index=True) if retained else pd.DataFrame(columns=["patientunitstayid", "diagnosisstring", "icd9code"])
    dx["text"] = (dx["diagnosisstring"].fillna("") + " " + dx["icd9code"].fillna("")).str.lower()
    mappings = {
        "chf": r"congestive heart failure|\bheart failure\b",
        "hypertension": r"hypertension|hypertensive",
        "dm": r"diabetes|\bdm\b",
        "ckd": r"chronic kidney|chronic renal|end stage renal|\besrd\b",
        "copd": r"chronic obstructive|\bcopd\b|emphysema",
        "liver": r"cirrhosis|chronic liver|hepatic failure",
        "cancer": r"malignan|\bcancer\b|neoplasm|carcinoma|metastat",
        "pvd": r"peripheral vascular|peripheral artery|\bpvd\b",
        "stroke": r"\bstroke\b|cerebrovascular accident|\bcva\b",
        "mi": r"myocardial infarction|\bmi\b",
        "obesity": r"obesity|obese",
        "anemia": r"\banemia\b|anaemia",
    }
    flags = pd.DataFrame({"stay_id": index["stay_id"].astype(int)})
    for name, pattern in mappings.items():
        matched = dx.loc[dx["text"].str.contains(pattern, regex=True, na=False), "patientunitstayid"].dropna().astype(int).unique()
        flags[name] = flags["stay_id"].isin(matched).astype(int)
    # A transparent approximation for adjustment/prediction transport. It is
    # not reported as a native eICU Charlson score.
    flags["charlson_score"] = flags[list(mappings)].sum(axis=1)
    return flags


def filter_eicu_labs(index: pd.DataFrame) -> pd.DataFrame:
    cache = CACHE_DIR / "eicu_selected_labs.csv.gz"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    ids = set(index["stay_id"].astype(int))
    target_names = {"creatinine", "BUN", "Hgb", "lactate", "WBC x 1000", "platelets x 1000", "potassium", "sodium"}
    retained = []
    for i, chunk in enumerate(
        pd.read_csv(LAB_FILE, usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"], chunksize=CHUNK_SIZE, low_memory=False), 1
    ):
        chunk["patientunitstayid"] = pd.to_numeric(chunk["patientunitstayid"], errors="coerce")
        chunk["labresultoffset"] = pd.to_numeric(chunk["labresultoffset"], errors="coerce")
        sub = chunk.loc[
            chunk["patientunitstayid"].isin(ids)
            & chunk["labname"].isin(target_names)
            & chunk["labresultoffset"].between(-PRE_LOOKBACK_MINUTES, FOLLOWUP_MINUTES)
        ].copy()
        if not sub.empty:
            retained.append(sub)
        if i % 10 == 0:
            print(f"Scanned {i * CHUNK_SIZE:,} eICU laboratory rows...", flush=True)
    labs = pd.concat(retained, ignore_index=True) if retained else pd.DataFrame(columns=["patientunitstayid", "labresultoffset", "labname", "labresult"])
    labs = labs.rename(columns={"patientunitstayid": "stay_id", "labresultoffset": "offset_min", "labresult": "value"})
    labs["stay_id"] = labs["stay_id"].astype(int)
    labs["offset_min"] = pd.to_numeric(labs["offset_min"], errors="coerce")
    labs["value"] = pd.to_numeric(labs["value"], errors="coerce")
    labs = labs.dropna(subset=["offset_min", "value"]).sort_values(["stay_id", "labname", "offset_min"])
    labs.to_csv(cache, index=False, compression="gzip")
    return labs


def filter_eicu_vitals(index: pd.DataFrame) -> pd.DataFrame:
    cache = CACHE_DIR / "eicu_selected_vitals.csv.gz"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    ids = set(index["stay_id"].astype(int))
    retained = []
    cols = ["patientunitstayid", "observationoffset", "sao2", "heartrate", "systemicsystolic", "systemicmean"]
    for i, chunk in enumerate(pd.read_csv(VITAL_FILE, usecols=cols, chunksize=CHUNK_SIZE, low_memory=False), 1):
        chunk["patientunitstayid"] = pd.to_numeric(chunk["patientunitstayid"], errors="coerce")
        chunk["observationoffset"] = pd.to_numeric(chunk["observationoffset"], errors="coerce")
        sub = chunk.loc[
            chunk["patientunitstayid"].isin(ids)
            & chunk["observationoffset"].gt(0)
            & chunk["observationoffset"].le(PRE24_MINUTES)
        ].copy()
        if not sub.empty:
            retained.append(sub)
        if i % 10 == 0:
            print(f"Scanned {i * CHUNK_SIZE:,} eICU periodic-vital rows...", flush=True)
    vitals = pd.concat(retained, ignore_index=True) if retained else pd.DataFrame(columns=cols)
    vitals = vitals.rename(columns={"patientunitstayid": "stay_id", "observationoffset": "offset_min"})
    vitals["stay_id"] = pd.to_numeric(vitals["stay_id"], errors="coerce").astype("Int64")
    vitals["offset_min"] = pd.to_numeric(vitals["offset_min"], errors="coerce")
    for col in ["sao2", "heartrate", "systemicsystolic", "systemicmean"]:
        vitals[col] = pd.to_numeric(vitals[col], errors="coerce")
    vitals = vitals.dropna(subset=["stay_id", "offset_min"])
    vitals.to_csv(cache, index=False, compression="gzip")
    return vitals


def last_value(frame: pd.DataFrame, name: str, start: float, end: float) -> pd.Series:
    sub = frame.loc[(frame["labname"] == name) & frame["offset_min"].between(start, end)].sort_values(["stay_id", "offset_min"])
    return sub.groupby("stay_id", sort=False)["value"].last()


def lab_window_features(labs: pd.DataFrame, start: float, end: float, suffix: str) -> pd.DataFrame:
    name_map = {
        "bun": "BUN", "creatinine": "creatinine", "hemoglobin": "Hgb", "lactate": "lactate",
        "wbc": "WBC x 1000", "platelet": "platelets x 1000", "potassium": "potassium", "sodium": "sodium",
    }
    work = labs.loc[labs["offset_min"].gt(start) & labs["offset_min"].le(end)].copy()
    result = pd.DataFrame(index=pd.Index([], name="stay_id"))
    if work.empty:
        return result
    for out_name, eicu_name in name_map.items():
        sub = work.loc[work["labname"].eq(eicu_name)].sort_values(["stay_id", "offset_min"])
        if sub.empty:
            continue
        grouped = sub.groupby("stay_id", sort=False)["value"]
        result[f"lab_{suffix}_{out_name}_last"] = grouped.last()
        result[f"lab_{suffix}_{out_name}_min"] = grouped.min()
        result[f"lab_{suffix}_{out_name}_max"] = grouped.max()
    return result


def preindex_lab_features(labs: pd.DataFrame) -> pd.DataFrame:
    name_map = {
        "bun": "BUN", "creatinine": "creatinine", "hemoglobin": "Hgb", "lactate": "lactate",
        "wbc": "WBC x 1000", "platelet": "platelets x 1000", "potassium": "potassium", "sodium": "sodium",
    }
    work = labs.loc[labs["offset_min"].between(-PRE24_MINUTES, 0)].copy()
    result = pd.DataFrame(index=pd.Index([], name="stay_id"))
    for out_name, eicu_name in name_map.items():
        sub = work.loc[work["labname"].eq(eicu_name)].sort_values(["stay_id", "offset_min"])
        if not sub.empty:
            result[f"lab_pre24h_{out_name}_last"] = sub.groupby("stay_id", sort=False)["value"].last()
    return result


def vital_window_features(vitals: pd.DataFrame, end: float, suffix: str) -> pd.DataFrame:
    colmap = {"map": "systemicmean", "heart_rate": "heartrate", "sbp": "systemicsystolic", "spo2": "sao2"}
    work = vitals.loc[vitals["offset_min"].le(end)].copy()
    result = pd.DataFrame(index=pd.Index([], name="stay_id"))
    for out_name, source in colmap.items():
        sub = work[["stay_id", "offset_min", source]].dropna().sort_values(["stay_id", "offset_min"])
        if sub.empty:
            continue
        grouped = sub.groupby("stay_id", sort=False)[source]
        result[f"vital_{suffix}_{out_name}_last"] = grouped.last()
        result[f"vital_{suffix}_{out_name}_min"] = grouped.min()
        result[f"vital_{suffix}_{out_name}_max"] = grouped.max()
    return result


def rolling_rise(times: np.ndarray, values: np.ndarray, index: float = 0.0) -> float | None:
    window: deque[tuple[float, float]] = deque()
    for time, value in zip(times, values):
        while window and time - window[0][0] > 48 * 60:
            window.popleft()
        if time > index and window and value - min(v for _, v in window) >= 0.3 - 1e-12:
            return float(time)
        window.append((float(time), float(value)))
    return None


def derive_aki(index: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    scr = labs.loc[labs["labname"].eq("creatinine")].copy().sort_values(["stay_id", "offset_min"])
    records = []
    for n, (stay_id, group) in enumerate(scr.groupby("stay_id", sort=False), 1):
        group = group.dropna(subset=["value", "offset_min"]).sort_values("offset_min")
        pre = group.loc[group["offset_min"].between(-PRE_LOOKBACK_MINUTES, 0)].copy()
        strict_pre = group.loc[group["offset_min"].between(-PRE_LOOKBACK_MINUTES, -1e-9)].copy()
        if not strict_pre.empty:
            base_row = strict_pre.loc[strict_pre["value"].idxmin()]
            source = "lowest_scr_7d_pre_icu"
        elif not pre.empty:
            base_row = pre.sort_values("offset_min").iloc[0]
            source = "icu_admission_scr_offset0_fallback"
        else:
            records.append({"stay_id": int(stay_id), "incident_aki_evaluable": False, "ineligibility_reason": "baseline_scr_missing", "aki_final": np.nan})
            continue
        baseline = float(base_row["value"])
        # Pre-index AKI check follows the original v3 operational definition.
        pre_abs = rolling_rise(pre["offset_min"].to_numpy(), pre["value"].to_numpy(), index=-np.inf)
        pre_ratio = pre.loc[pre["value"].ge(1.5 * baseline), "offset_min"].min() if not pre.empty else np.nan
        preexisting = pre_abs is not None or pd.notna(pre_ratio)
        post = group.loc[group["offset_min"].gt(0) & group["offset_min"].le(FOLLOWUP_MINUTES)].copy()
        row = {
            "stay_id": int(stay_id), "baseline_scr_at_landmark": baseline,
            "baseline_scr_source": source, "baseline_scr_time_offset_min": float(base_row["offset_min"]),
            "preexisting_aki_at_or_before_index": bool(preexisting), "post_index_scr_n_7d": int(len(post)),
            "incident_aki_evaluable": False, "ineligibility_reason": "", "aki_final": np.nan,
            "aki_onset_hours_final": np.nan, "aki_stage_final": np.nan,
        }
        if preexisting:
            row["ineligibility_reason"] = "aki_present_before_or_at_index"
        elif post.empty:
            row["ineligibility_reason"] = "no_post_index_scr_within_7d"
        else:
            absolute = rolling_rise(group.loc[group["offset_min"].between(-48 * 60, FOLLOWUP_MINUTES), "offset_min"].to_numpy(), group.loc[group["offset_min"].between(-48 * 60, FOLLOWUP_MINUTES), "value"].to_numpy())
            ratios = post.loc[post["value"].ge(1.5 * baseline), "offset_min"]
            ratio_time = float(ratios.min()) if not ratios.empty else None
            onset_options = [x for x in (absolute, ratio_time) if x is not None and np.isfinite(x)]
            peak = float(post["value"].max())
            peak_ratio = peak / baseline if baseline > 0 else np.nan
            aki = bool(onset_options)
            stage = 0
            if aki:
                if peak_ratio >= 3 or peak >= 4:
                    stage = 3
                elif peak_ratio >= 2:
                    stage = 2
                else:
                    stage = 1
            row.update({
                "incident_aki_evaluable": True, "ineligibility_reason": "eligible", "aki_final": int(aki),
                "aki_onset_hours_final": min(onset_options) / 60 if onset_options else np.nan,
                "aki_stage_final": stage, "post_index_peak_scr_7d": peak, "peak_scr_ratio_to_baseline": peak_ratio,
            })
        records.append(row)
        if n % 10_000 == 0:
            print(f"Derived eICU AKI outcome for {n:,} stays...", flush=True)
    return pd.DataFrame(records)


def build_external_cohort() -> pd.DataFrame:
    cache = CACHE_DIR / "cohort_v16_eicu_harmonized.csv.gz"
    if cache.exists():
        cohort = pd.read_csv(cache, low_memory=False)
        cohort["incident_aki_evaluable"] = cohort["incident_aki_evaluable"].fillna(False)
        cohort["ineligibility_reason"] = cohort["ineligibility_reason"].fillna("baseline_scr_missing")
        cohort["baseline_scr_source"] = cohort["baseline_scr_source"].fillna("missing")
        return cohort
    index = read_eicu_surgical_index()
    print(f"Strict eICU surgical index cohort: {len(index):,} first ICU stays.", flush=True)
    dx = extract_eicu_diagnoses(index)
    labs = filter_eicu_labs(index)
    print(f"Retained {len(labs):,} relevant laboratory rows for external cohort.", flush=True)
    vitals = filter_eicu_vitals(index)
    print(f"Retained {len(vitals):,} periodic-vital rows through 24 h for external cohort.", flush=True)
    aki = derive_aki(index, labs)
    cohort = index.merge(dx, on="stay_id", how="left", validate="one_to_one").merge(aki, on="stay_id", how="left", validate="one_to_one")
    # Stays with no creatinine row do not enter derive_aki; label them
    # explicitly rather than leaving a misleading unassessed outcome state.
    cohort["incident_aki_evaluable"] = cohort["incident_aki_evaluable"].fillna(False)
    cohort["ineligibility_reason"] = cohort["ineligibility_reason"].fillna("baseline_scr_missing")
    cohort["baseline_scr_source"] = cohort["baseline_scr_source"].fillna("missing")
    cohort = cohort.set_index("stay_id", drop=False)
    for frame in [preindex_lab_features(labs), lab_window_features(labs, 0, 6 * 60, "0_6h"), lab_window_features(labs, 0, 24 * 60, "0_24h"), vital_window_features(vitals, 6 * 60, "0_6h"), vital_window_features(vitals, 24 * 60, "0_24h")]:
        cohort = cohort.join(frame, how="left")
    cohort = cohort.reset_index(drop=True)
    cohort.to_csv(cache, index=False, compression="gzip")
    return cohort


def portable_predictors(landmark: int) -> list[str]:
    predictors = list(PORTABLE_0H)
    if landmark in (6, 24):
        suffix = "0_6h" if landmark == 6 else "0_24h"
        predictors += [f"lab_{suffix}_{base}_{stat}" for base in LAB_BASES for stat in ("last", "min", "max")]
        predictors += [f"vital_{suffix}_{base}_{stat}" for base in VITAL_BASES for stat in ("last", "min", "max")]
    return predictors


def make_model(landmark: int, continuous: list[str], binary: list[str], categorical: list[str]) -> Pipeline:
    if landmark == 24:
        return Pipeline([
            ("preprocess", make_preprocessor_custom(continuous, binary, categorical, scale=True, add_missing_indicators=True)),
            ("model", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE)),
        ])
    return Pipeline([
        ("preprocess", make_preprocessor_custom(continuous, binary, categorical, scale=False, add_missing_indicators=True)),
        ("model", xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=4, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary:logistic",
            eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y, dtype=int), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    predicted = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "n": int(len(y)), "event_n": int(y.sum()), "event_rate": float(y.mean()),
        "auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p), "brier_score": brier_score_loss(y, p),
        "accuracy_0_5": accuracy_score(y, predicted), "sensitivity_0_5": recall_score(y, predicted, zero_division=0),
        "specificity_0_5": float(tn / (tn + fp)) if tn + fp else np.nan,
        "precision_0_5": precision_score(y, predicted, zero_division=0), "f1_0_5": f1_score(y, predicted, zero_division=0),
        "calibration_intercept": intercept, "calibration_slope": slope,
        "mean_predicted_risk": float(p.mean()), "observed_expected_ratio": float(y.mean() / p.mean()) if p.mean() else np.nan,
    }


def bootstrap_auroc(y: np.ndarray, p: np.ndarray, draws: int = 300) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    values = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            values.append(roc_auc_score(y[idx], p[idx]))
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def eicu_landmark_dataset(cohort: pd.DataFrame, landmark: int, predictors: list[str]) -> pd.DataFrame:
    work = cohort.loc[cohort["incident_aki_evaluable"].fillna(False).astype(bool)].copy()
    if landmark > 0:
        early = pd.to_numeric(work["aki_onset_hours_final"], errors="coerce").le(landmark) & work["aki_final"].eq(1)
        work = work.loc[~early].copy()
        work[OUTCOME] = ((work["aki_final"].eq(1)) & (pd.to_numeric(work["aki_onset_hours_final"], errors="coerce") > landmark)).astype(int)
    else:
        work[OUTCOME] = work["aki_final"].astype(int)
    for col in predictors:
        if col not in work.columns:
            work[col] = np.nan
    return work


def fit_and_evaluate(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows, prediction_tables, mapping_rows = [], [], []
    for landmark in (0, 6, 24):
        predictors = portable_predictors(landmark)
        dev = load_data(landmark).copy()
        available = [p for p in predictors if p in dev.columns]
        missing_in_dev = sorted(set(predictors) - set(available))
        if missing_in_dev:
            raise ValueError(f"Portable predictors unavailable in MIMIC {landmark}h: {missing_in_dev}")
        ext = eicu_landmark_dataset(cohort, landmark, predictors)
        continuous, binary, categorical = identify_types_for_columns(dev[available], available)
        train_subjects, test_subjects, _ = choose_grouped_split(dev)
        train = dev.loc[dev["subject_id"].astype(int).isin(train_subjects)].copy()
        test = dev.loc[dev["subject_id"].astype(int).isin(test_subjects)].copy()
        reference_model = make_model(landmark, continuous, binary, categorical)
        reference_model.fit(train[available], train[OUTCOME].astype(int).to_numpy())
        p_internal = reference_model.predict_proba(test[available])[:, 1]
        model = make_model(landmark, continuous, binary, categorical)
        model.fit(dev[available], dev[OUTCOME].astype(int).to_numpy())
        p_external = model.predict_proba(ext[available])[:, 1]
        for label, data, p in [("MIMIC grouped held-out reference", test, p_internal), ("eICU external validation", ext, p_external)]:
            y = data[OUTCOME].astype(int).to_numpy()
            row = {"landmark_hours": landmark, "model_family": "Logistic Regression" if landmark == 24 else "XGBoost", "evaluation_dataset": label, "predictor_n": len(available)}
            row.update(metrics(y, p))
            low, high = bootstrap_auroc(y, p)
            row.update({"auroc_ci_lower": low, "auroc_ci_upper": high})
            summary_rows.append(row)
        pred = ext[["stay_id", "subject_id", "hospitalid", "operative_system", OUTCOME]].copy()
        pred["landmark_hours"] = landmark
        pred["predicted_risk_portable_model"] = p_external
        prediction_tables.append(pred)
        for col in available:
            mapping_rows.append({
                "landmark_hours": landmark, "predictor": col,
                "mimic_source": col,
                "eicu_mapping": eicu_mapping_text(col),
                "external_missing_percent": float(ext[col].isna().mean() * 100),
            })
    summary = pd.DataFrame(summary_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    mapping = pd.DataFrame(mapping_rows)
    return summary, predictions, mapping


def eicu_mapping_text(col: str) -> str:
    if col in {"gender", "anchor_age"}:
        return "eICU patient table, normalized to MIMIC-compatible coding"
    if col in {"cardiac_surgery", "non_cardiac_surgery", "general_gi_hepatobiliary_surgery", "orthopedic_major_surgery", "neurosurgery", "thoracic_respiratory_surgery"}:
        return "eICU admissionDx operative organ-system category; cardiovascular mapped to cardiac; vascular-specific surgery unavailable"
    if col == "baseline_scr_at_landmark":
        return "lowest creatinine in 7 days before ICU, or creatinine at ICU offset 0 if no pre-ICU value"
    if col.startswith("lab_pre24h_") or col.startswith("lab_0_"):
        return "eICU lab table: BUN, creatinine, Hgb, lactate, WBC x 1000, platelets x 1000, potassium, sodium; same restricted time window"
    if col.startswith("vital_0_"):
        return "eICU vitalPeriodic table: systemic mean BP, heart rate, systemic systolic BP, SaO2; same restricted time window"
    if col == "charlson_score":
        return "sum of broad text/ICD-derived comorbidity indicators; approximate transport variable"
    return "eICU diagnosis text/ICD field; broad keyword mapping"


def cohort_audit(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({"metric": "strict_surgical_first_icu_stays", "value": len(cohort), "note": "Adults with eICU Operative Organ Systems diagnosis in five prespecified surgical systems"})
    rows.append({"metric": "unique_eicu_icu_stays", "value": int(cohort["stay_id"].nunique()), "note": "Must equal strict cohort row count after first-stay selection"})
    rows.append({"metric": "unique_eicu_patients", "value": int(cohort["subject_id"].nunique()), "note": "eICU uniquepid; may contribute more than one hospital stay"})
    rows.append({"metric": "participating_hospitals", "value": int(pd.to_numeric(cohort["hospitalid"], errors="coerce").nunique()), "note": "Multi-centre external validation coverage"})
    evaluable = cohort["incident_aki_evaluable"].fillna(False).astype(bool)
    rows.append({"metric": "incident_aki_evaluable", "value": int(evaluable.sum()), "note": "Requires baseline and >=1 post-index creatinine, excluding pre-index AKI"})
    rows.append({"metric": "incident_aki_evaluable_percent", "value": float(evaluable.mean() * 100), "note": "Percentage of strict surgical first-ICU cohort"})
    rows.append({"metric": "evaluable_aki_rate", "value": float(pd.to_numeric(cohort.loc[evaluable, "aki_final"], errors="coerce").mean()), "note": "SCr-only KDIGO incident AKI within 7 days"})
    for reason, count in cohort["ineligibility_reason"].fillna("not_assessed").value_counts().items():
        rows.append({"metric": f"outcome_status::{reason}", "value": int(count), "note": "External outcome eligibility audit"})
    for source, count in cohort["baseline_scr_source"].fillna("missing").value_counts().items():
        rows.append({"metric": f"baseline_source::{source}", "value": int(count), "note": "External AKI derivation"})
    for system, count in cohort["operative_system"].value_counts().items():
        rows.append({"metric": f"operative_system::{system}", "value": int(count), "note": "Strict cohort surgical classification"})
    return pd.DataFrame(rows)


def make_figures(performance: pd.DataFrame, predictions: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), dpi=180)
    for ax, landmark in zip(axes, (0, 6, 24)):
        group = predictions.loc[predictions["landmark_hours"].eq(landmark)]
        y, p = group[OUTCOME].astype(int), group["predicted_risk_portable_model"]
        fpr, tpr, _ = roc_curve(y, p)
        auc = performance.loc[(performance["landmark_hours"] == landmark) & (performance["evaluation_dataset"] == "eICU external validation"), "auroc"].iloc[0]
        ax.plot(fpr, tpr, color="#4c78a8", linewidth=2, label=f"AUROC {auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="#687080", linewidth=1)
        ax.set_title(f"{landmark} h")
        ax.set_xlabel("1 − specificity")
        ax.legend(frameon=False, loc="lower right")
    axes[0].set_ylabel("Sensitivity")
    fig.suptitle("Feature-harmonized external validation in eICU", y=1.03)
    fig.text(0.5, .965, "SCr-based postoperative incident AKI; frozen MIMIC-trained portable models", ha="center", fontsize=9, color="#5f6675")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUTPUT_DIR / "figure_v16_eicu_external_roc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), dpi=180, sharey=True)
    for ax, landmark in zip(axes, (0, 6, 24)):
        group = predictions.loc[predictions["landmark_hours"].eq(landmark)].copy()
        group["bin"] = pd.qcut(group["predicted_risk_portable_model"], q=min(10, group["predicted_risk_portable_model"].nunique()), duplicates="drop")
        calibration = group.groupby("bin", observed=False).agg(pred=("predicted_risk_portable_model", "mean"), obs=(OUTCOME, "mean"), n=(OUTCOME, "size"))
        ax.plot([0, 1], [0, 1], "--", color="#687080", linewidth=1)
        ax.plot(calibration["pred"], calibration["obs"], marker="o", color="#e17c05")
        ax.set_title(f"{landmark} h")
        ax.set_xlabel("Mean predicted risk")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed AKI risk")
    fig.suptitle("Calibration of frozen portable models in eICU", y=1.03)
    fig.text(0.5, .965, "Decile calibration; no eICU recalibration applied", ha="center", fontsize=9, color="#5f6675")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUTPUT_DIR / "figure_v16_eicu_external_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_readme(cohort: pd.DataFrame, audit: pd.DataFrame, performance: pd.DataFrame) -> None:
    ext = performance.loc[performance["evaluation_dataset"].eq("eICU external validation")]
    lines = ["# v16 eICU-CRD external validation", "", "## Design", ""]
    lines.append("This is a feature-harmonized multicentre external validation in eICU-CRD v2.0. The MIMIC-IV primary models and all primary results remain unchanged.")
    lines.append("")
    lines.append("Adults were selected when eICU admissionDx recorded an Operative Organ Systems diagnosis in cardiovascular, gastrointestinal, neurologic, respiratory, or musculoskeletal/skin systems. One first ICU stay per eICU hospital stay was retained.")
    lines.append("")
    lines.append("Because eICU lacks procedure codes and does not identically encode all MIMIC predictors, frozen portable models were refit on all MIMIC modelling-ready development data using only explicitly harmonized variables. eICU was not used for model fitting, tuning, feature selection, or recalibration.")
    lines.append("")
    lines.extend(["## AKI outcome", ""])
    lines.append("The external outcome uses the same serum-creatinine KDIGO operational rule: rise >=0.3 mg/dL within 48 h or >=1.5× baseline within 7 days after ICU admission. Baseline is the lowest eICU creatinine in the preceding 7 days, with an ICU-offset-0 creatinine fallback when a pre-ICU result was unavailable. Urine output was not used.")
    evaluable = cohort["incident_aki_evaluable"].fillna(False).astype(bool)
    hospital_n = pd.to_numeric(cohort["hospitalid"], errors="coerce").nunique()
    lines.append(f"Of {len(cohort):,} strict surgical first ICU stays across {hospital_n:,} hospitals, {int(evaluable.sum()):,} ({evaluable.mean()*100:.1f}%) were outcome-evaluable.")
    lines.append("")
    lines.extend(["## External performance", ""])
    for r in ext.itertuples():
        lines.append(f"- {int(r.landmark_hours)} h ({r.model_family}): n={int(r.n):,}, events={int(r.event_n):,} ({r.event_rate*100:.1f}%), AUROC {r.auroc:.3f} (95% CI {r.auroc_ci_lower:.3f}–{r.auroc_ci_upper:.3f}), AUPRC {r.auprc:.3f}, Brier {r.brier_score:.3f}, calibration slope {r.calibration_slope:.2f}.")
    lines.append("")
    lines.extend(["## Required interpretation caveats", ""])
    lines.extend([
        "- This validates harmonized portable versions, not the exact 36/72-predictor locked primary models.",
        "- Cardiovascular operative diagnoses were mapped to cardiac surgery because eICU admissionDx does not reliably separate cardiac from vascular procedures; vascular-specific surgery was unavailable.",
        "- eICU comorbidity flags and the approximate Charlson variable use broad diagnosis text/ICD mappings and may differ from MIMIC-derived comorbidity phenotypes.",
        "- The ICU-offset-0 baseline-creatinine fallback should be retained as a baseline-definition sensitivity analysis.",
    ])
    (OUTPUT_DIR / "audit_v16_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not EICU_ROOT.exists():
        raise FileNotFoundError(f"eICU root is unavailable: {EICU_ROOT}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cohort = build_external_cohort()
    cohort.to_csv(OUTPUT_DIR / "cohort_v16_eicu_external_validation.csv.gz", index=False, compression="gzip")
    audit = cohort_audit(cohort)
    audit.to_csv(OUTPUT_DIR / "audit_v16_eicu_cohort_summary.csv", index=False)
    performance, predictions, mapping = fit_and_evaluate(cohort)
    performance.to_csv(OUTPUT_DIR / "model_v16_portable_external_validation_performance.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "model_v16_portable_external_predictions.csv", index=False)
    mapping.to_csv(OUTPUT_DIR / "audit_v16_feature_harmonization.csv", index=False)
    make_figures(performance, predictions)
    write_readme(cohort, audit, performance)
    print(f"Wrote eICU external-validation outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
