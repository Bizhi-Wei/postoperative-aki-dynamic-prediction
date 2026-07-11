"""Build leakage-controlled 0 h, 6 h, and 24 h postoperative AKI datasets.

No machine learning is performed. Predictor columns are explicitly allowlisted.
Timestamped laboratory and ICU vital-sign features are recalculated within each
landmark window; legacy post_* and untimed laboratory summaries are not reused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIMIC_ROOT = PROJECT_ROOT.parent
INPUT_FILE = PROJECT_ROOT / "outputs" / "finalized_v3_1" / "cohort_v3_1_strict_main_evaluable.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dynamic_v4"
CACHE_DIR = PROJECT_ROOT / "outputs" / "cache_v4"
LABEVENTS = MIMIC_ROOT / "hosp" / "labevents.csv"
CHARTEVENTS = MIMIC_ROOT / "icu" / "chartevents.csv"

LAB_CHUNK_SIZE = 2_000_000
CHART_CHUNK_SIZE = 2_000_000

LAB_ITEM_TO_FEATURE = {
    50912: "creatinine", 52546: "creatinine",
    50813: "lactate", 52442: "lactate", 53154: "lactate",
    50882: "bicarbonate",
    50809: "glucose", 50931: "glucose", 52027: "glucose", 52569: "glucose",
    50822: "potassium", 50971: "potassium", 52452: "potassium", 52610: "potassium",
    50824: "sodium", 50983: "sodium", 52455: "sodium", 52623: "sodium",
    51006: "bun", 52647: "bun",
    51222: "hemoglobin",
    51237: "inr", 51675: "inr",
    51265: "platelet", 53189: "platelet",
    51301: "wbc", 51755: "wbc", 51756: "wbc",
    50820: "ph", 50821: "pao2", 50818: "paco2",
}

CHART_ITEM_TO_FEATURE = {
    220045: "heart_rate",
    220179: "sbp", 220050: "sbp",
    220180: "dbp", 220051: "dbp",
    220181: "map", 220052: "map",
    220210: "respiratory_rate", 224690: "respiratory_rate",
    220277: "spo2",
    223762: "temperature_c", 223761: "temperature_f",
}

PLAUSIBLE_RANGES = {
    "creatinine": (0.1, 50), "lactate": (0, 50), "bicarbonate": (2, 60),
    "glucose": (20, 1500), "potassium": (1, 10), "sodium": (80, 200),
    "bun": (1, 300), "hemoglobin": (2, 25), "inr": (0.5, 20),
    "platelet": (1, 2000), "wbc": (0.1, 500), "ph": (6.5, 8.0),
    "pao2": (10, 800), "paco2": (5, 200), "heart_rate": (20, 250),
    "sbp": (30, 300), "dbp": (10, 200), "map": (20, 250),
    "respiratory_rate": (3, 80), "spo2": (20, 100), "temperature_c": (25, 45),
}

IDENTIFIER_COLUMNS = ["subject_id", "hadm_id", "stay_id", "intime", "index_surgery_date"]

STATIC_PREDICTOR_ALLOWLIST = [
    "first_careunit", "gender", "anchor_age", "race", "admission_type",
    "insurance", "marital_status", "chf", "hypertension", "dm", "dm_comp",
    "ckd", "copd", "liver", "cancer", "pvd", "stroke", "mi", "obesity",
    "anemia", "charlson_score", "surgery_categories", "n_qualifying_codes",
    "days_from_procedure_to_icu", "cardiac_surgery", "non_cardiac_surgery",
    "vascular_surgery", "general_gi_hepatobiliary_surgery",
    "orthopedic_major_surgery", "neurosurgery", "thoracic_respiratory_surgery",
]

EXPLICIT_LEAKAGE_COLUMNS = {
    "outtime", "dischtime", "dod", "los", "hosp_los", "icu_death", "hosp_death",
    "death_90d", "death_365d", "hospital_expire_flag", "aki_provisional",
    "aki_stage_provisional", "aki_onset_time_provisional", "aki_onset_days_provisional",
    "aki_final", "aki_stage_final", "aki_onset_time_final", "aki_onset_hours_final",
    "aki_criterion_0_3_within_48h", "aki_criterion_1_5x_within_7d",
    "peak_scr_ratio_to_baseline", "post_index_scr_n_7d", "post_index_first_scr_time",
    "post_index_peak_scr_7d", "post_index_peak_scr_time", "aki_label_comparable",
    "aki_label_discordant", "incident_aki_ineligibility_reason",
    "preexisting_aki_first_time", "preexisting_aki_at_or_before_index",
}


def require_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"Input cohort is missing required columns: {missing}")


def bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.lower().isin(["true", "1", "yes"])


def load_cohort() -> pd.DataFrame:
    data = pd.read_csv(INPUT_FILE, low_memory=False)
    require_columns(
        data,
        [
            *IDENTIFIER_COLUMNS, "aki_final", "aki_onset_hours_final",
            "incident_aki_evaluable", "baseline_scr_final", "baseline_scr_time",
            "baseline_scr_source", *STATIC_PREDICTOR_ALLOWLIST,
        ],
    )
    data["intime"] = pd.to_datetime(data["intime"], errors="coerce")
    data["index_surgery_date"] = pd.to_datetime(data["index_surgery_date"], errors="coerce")
    data["baseline_scr_time"] = pd.to_datetime(data["baseline_scr_time"], errors="coerce")
    if not bool_mask(data["incident_aki_evaluable"]).all():
        raise ValueError("v4 input contains non-evaluable incident-AKI rows")
    if data["hadm_id"].duplicated().any() or data["stay_id"].duplicated().any():
        raise ValueError("v4 input is not unique by hadm_id and stay_id")
    return data


def clean_feature_values(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned.loc[cleaned["feature"].eq("temperature_f"), "valuenum"] = (
        cleaned.loc[cleaned["feature"].eq("temperature_f"), "valuenum"] - 32
    ) * 5 / 9
    cleaned.loc[cleaned["feature"].eq("temperature_f"), "feature"] = "temperature_c"
    valid = pd.Series(False, index=cleaned.index)
    for feature, (lower, upper) in PLAUSIBLE_RANGES.items():
        valid |= cleaned["feature"].eq(feature) & cleaned["valuenum"].between(lower, upper)
    return cleaned.loc[valid].copy()


def extract_labs(cohort: pd.DataFrame) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "aligned_dynamic_labs_pre24_to_post24.csv.gz"
    if cache.exists():
        print(f"Loading cached dynamic labs: {cache}", flush=True)
        return pd.read_csv(cache, parse_dates=["charttime"], low_memory=False)
    retained: list[pd.DataFrame] = []
    index = cohort[["subject_id", "stay_id", "intime"]]
    subject_ids = set(index["subject_id"].astype(int))
    usecols = ["labevent_id", "subject_id", "itemid", "charttime", "valuenum"]
    valid_itemids = set(LAB_ITEM_TO_FEATURE)
    for chunk_no, chunk in enumerate(
        pd.read_csv(LABEVENTS, usecols=usecols, chunksize=LAB_CHUNK_SIZE, low_memory=False), start=1
    ):
        selected = chunk.loc[
            chunk["itemid"].isin(valid_itemids)
            & chunk["subject_id"].isin(subject_ids)
            & chunk["valuenum"].notna()
        ].copy()
        if not selected.empty:
            selected["charttime"] = pd.to_datetime(selected["charttime"], errors="coerce")
            selected["feature"] = selected["itemid"].map(LAB_ITEM_TO_FEATURE)
            selected = selected.dropna(subset=["charttime"]).merge(
                index, on="subject_id", how="inner", validate="many_to_many"
            )
            selected["hours_from_icu"] = (
                selected["charttime"] - selected["intime"]
            ).dt.total_seconds() / 3600
            retained.append(
                selected.loc[selected["hours_from_icu"].between(-24, 24, inclusive="both")]
            )
        if chunk_no % 10 == 0:
            print(f"Scanned {chunk_no * LAB_CHUNK_SIZE:,} labevents rows...", flush=True)
    labs = clean_feature_values(pd.concat(retained, ignore_index=True))
    labs.to_csv(cache, index=False, compression="gzip")
    return labs


def extract_vitals(cohort: pd.DataFrame) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "aligned_dynamic_vitals_post24.csv.gz"
    if cache.exists():
        print(f"Loading cached dynamic vitals: {cache}", flush=True)
        return pd.read_csv(cache, parse_dates=["charttime"], low_memory=False)
    retained: list[pd.DataFrame] = []
    index = cohort[["stay_id", "intime"]]
    stay_ids = set(index["stay_id"].astype(int))
    usecols = ["stay_id", "itemid", "charttime", "valuenum"]
    valid_itemids = set(CHART_ITEM_TO_FEATURE)
    for chunk_no, chunk in enumerate(
        pd.read_csv(CHARTEVENTS, usecols=usecols, chunksize=CHART_CHUNK_SIZE, low_memory=False), start=1
    ):
        selected = chunk.loc[
            chunk["itemid"].isin(valid_itemids)
            & chunk["stay_id"].isin(stay_ids)
            & chunk["valuenum"].notna()
        ].copy()
        if not selected.empty:
            selected["charttime"] = pd.to_datetime(selected["charttime"], errors="coerce")
            selected["feature"] = selected["itemid"].map(CHART_ITEM_TO_FEATURE)
            selected = selected.dropna(subset=["charttime"]).merge(
                index, on="stay_id", how="inner", validate="many_to_one"
            )
            selected["hours_from_icu"] = (
                selected["charttime"] - selected["intime"]
            ).dt.total_seconds() / 3600
            retained.append(
                selected.loc[selected["hours_from_icu"].between(0, 24, inclusive="both")]
            )
        if chunk_no % 20 == 0:
            print(f"Scanned {chunk_no * CHART_CHUNK_SIZE:,} chartevents rows...", flush=True)
    vitals = clean_feature_values(pd.concat(retained, ignore_index=True))
    vitals.to_csv(cache, index=False, compression="gzip")
    return vitals


def align_labs_to_stays(labs: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    index = cohort[["subject_id", "stay_id", "intime"]]
    aligned = labs.merge(index, on="subject_id", how="inner", validate="many_to_many")
    aligned["hours_from_icu"] = (
        aligned["charttime"] - aligned["intime"]
    ).dt.total_seconds() / 3600
    return aligned.loc[aligned["hours_from_icu"].between(-24, 24, inclusive="both")].copy()


def align_vitals_to_stays(vitals: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    index = cohort[["stay_id", "intime"]]
    aligned = vitals.merge(index, on="stay_id", how="inner", validate="many_to_one")
    aligned["hours_from_icu"] = (
        aligned["charttime"] - aligned["intime"]
    ).dt.total_seconds() / 3600
    return aligned.loc[aligned["hours_from_icu"].between(0, 24, inclusive="both")].copy()


def aggregate_last(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["stay_id"])
    last = (
        data.sort_values(["stay_id", "feature", "charttime"])
        .groupby(["stay_id", "feature"], as_index=False)
        .tail(1)
        .pivot(index="stay_id", columns="feature", values="valuenum")
    )
    last.columns = [f"{prefix}_{feature}_last" for feature in last.columns]
    return last.reset_index()


def aggregate_window(data: pd.DataFrame, landmark: int, source: str) -> pd.DataFrame:
    window = data.loc[
        data["hours_from_icu"].gt(0) & data["hours_from_icu"].le(landmark)
    ].sort_values(["stay_id", "feature", "charttime"])
    if window.empty:
        return pd.DataFrame(columns=["stay_id"])
    grouped = window.groupby(["stay_id", "feature"])["valuenum"]
    summary = grouped.agg(["min", "max", "last", "count"]).unstack("feature")
    summary.columns = [
        f"{source}_0_{landmark}h_{feature}_{stat}" for stat, feature in summary.columns
    ]
    return summary.reset_index()


def baseline_features_at_landmark(cohort: pd.DataFrame, landmark: int) -> pd.DataFrame:
    available = cohort["baseline_scr_time"].le(cohort["intime"] + pd.to_timedelta(landmark, unit="h"))
    result = pd.DataFrame({"stay_id": cohort["stay_id"]})
    result["baseline_scr_available_at_landmark"] = available
    result["baseline_scr_at_landmark"] = cohort["baseline_scr_final"].where(available)
    result["baseline_scr_source_at_landmark"] = cohort["baseline_scr_source"].where(available)
    result["baseline_to_icu_hours_at_landmark"] = (
        (cohort["intime"] - cohort["baseline_scr_time"]).dt.total_seconds() / 3600
    ).where(available)
    return result


def build_landmark_dataset(
    cohort: pd.DataFrame,
    preindex_labs: pd.DataFrame,
    labs_aligned: pd.DataFrame,
    vitals_aligned: pd.DataFrame,
    landmark: int,
) -> tuple[pd.DataFrame, int, list[str]]:
    aki = bool_mask(cohort["aki_final"])
    onset = pd.to_numeric(cohort["aki_onset_hours_final"], errors="coerce")
    early_aki = aki & onset.le(landmark) if landmark > 0 else pd.Series(False, index=cohort.index)
    eligible = cohort.loc[~early_aki].copy()

    metadata = eligible[IDENTIFIER_COLUMNS].copy()
    metadata["landmark_hours"] = landmark
    predictors = eligible[[column for column in STATIC_PREDICTOR_ALLOWLIST if column in eligible]].copy()
    predictors.insert(0, "stay_id", eligible["stay_id"].values)
    predictors = predictors.merge(
        baseline_features_at_landmark(eligible, landmark), on="stay_id", how="left", validate="one_to_one"
    )
    predictors = predictors.merge(preindex_labs, on="stay_id", how="left", validate="one_to_one")

    if landmark > 0:
        predictors = predictors.merge(
            aggregate_window(labs_aligned, landmark, "lab"),
            on="stay_id", how="left", validate="one_to_one",
        )
        predictors = predictors.merge(
            aggregate_window(vitals_aligned, landmark, "vital"),
            on="stay_id", how="left", validate="one_to_one",
        )

    outcome = pd.DataFrame({"stay_id": eligible["stay_id"]})
    outcome["outcome_aki_after_landmark_to_7d"] = bool_mask(eligible["aki_final"]).values
    outcome["outcome_aki_onset_hours_from_icu"] = eligible["aki_onset_hours_final"].where(
        bool_mask(eligible["aki_final"])
    ).values

    dataset = metadata.merge(predictors, on="stay_id", how="left", validate="one_to_one")
    dataset = dataset.merge(outcome, on="stay_id", how="left", validate="one_to_one")
    predictor_columns = [
        column for column in predictors.columns if column != "stay_id"
    ]
    return dataset, int(early_aki.sum()), predictor_columns


def leakage_exclusions(source_columns: Iterable[str]) -> list[str]:
    excluded = []
    old_untimed_labs = {
        "lactate", "paco2", "ph", "pao2", "bicarbonate", "scr", "glucose",
        "potassium", "sodium", "bun", "hemoglobin", "inr", "platelet", "wbc",
    }
    for column in source_columns:
        if column.startswith("post_") or column in EXPLICIT_LEAKAGE_COLUMNS or column in old_untimed_labs:
            excluded.append(column)
    return sorted(excluded)


def write_readme(
    summaries: pd.DataFrame,
    excluded: list[str],
    predictor_lists: dict[int, list[str]],
) -> None:
    rows = summaries.set_index("landmark_hours")
    content = f"""# Dynamic postoperative AKI datasets v4

## Design

- Source cohort: strict v3.1 main evaluable cohort.
- Analysis unit: first ICU stay per hospital admission.
- Landmarks: ICU admission (0 h), 6 h, and 24 h.
- Outcome: incident final KDIGO SCr AKI occurring after the landmark through 7 days.
- Patients with AKI onset at or before a positive-hour landmark are excluded from that landmark dataset.
- No urine-output criterion and no machine learning were used.

## Dataset sizes

| Landmark | N | Early AKI excluded | Future AKI n | Incidence | Predictors |
|---:|---:|---:|---:|---:|---:|
| 0 h | {int(rows.loc[0,'sample_size']):,} | {int(rows.loc[0,'excluded_aki_at_or_before_landmark_n']):,} | {int(rows.loc[0,'event_n']):,} | {rows.loc[0,'event_incidence_percent']:.2f}% | {int(rows.loc[0,'predictor_count']):,} |
| 6 h | {int(rows.loc[6,'sample_size']):,} | {int(rows.loc[6,'excluded_aki_at_or_before_landmark_n']):,} | {int(rows.loc[6,'event_n']):,} | {rows.loc[6,'event_incidence_percent']:.2f}% | {int(rows.loc[6,'predictor_count']):,} |
| 24 h | {int(rows.loc[24,'sample_size']):,} | {int(rows.loc[24,'excluded_aki_at_or_before_landmark_n']):,} | {int(rows.loc[24,'event_n']):,} | {rows.loc[24,'event_incidence_percent']:.2f}% | {int(rows.loc[24,'predictor_count']):,} |

## Predictor timing

- Static allowlist: demographics, admission descriptors, comorbidities, surgery categories, and ICU type.
- Baseline SCr is present only when `baseline_scr_time <= landmark_time`. Thus post-ICU admission fallback baselines are hidden at 0 h and become available only at later landmarks when actually charted.
- Pre-index labs: last valid result from -24 h through ICU admission.
- 6 h and 24 h labs/vitals: min, max, last, and count using only `(0, landmark]` observations.
- Laboratory features: creatinine, lactate, bicarbonate, glucose, potassium, sodium, BUN, hemoglobin, INR, platelets, WBC, pH, PaO2, PaCO2.
- Vital features: heart rate, SBP, DBP, MAP, respiratory rate, SpO2, temperature.

Candidate predictors are explicit in each dataset. The last two columns are outcomes and are not predictors: `outcome_aki_after_landmark_to_7d` and `outcome_aki_onset_hours_from_icu`.

## Leakage exclusions

The following {len(excluded)} source columns were explicitly excluded because they contain outcomes, post-index whole-period summaries, untimed laboratory summaries, mortality/LOS information, or AKI-definition information:

`{'`, `'.join(excluded)}`

## Reproducibility

The script caches filtered timestamped events under `outputs/cache_v4/`. It does not reuse legacy `post_*` summaries. No machine learning was performed.
"""
    (OUTPUT_DIR / "audit_v4_readme.md").write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort()
    labs = extract_labs(cohort)
    vitals = extract_vitals(cohort)
    print(f"Retained {len(labs):,} candidate lab rows and {len(vitals):,} vital rows.", flush=True)

    labs_aligned = labs
    vitals_aligned = vitals
    preindex_labs = aggregate_last(
        labs_aligned.loc[labs_aligned["hours_from_icu"].between(-24, 0, inclusive="both")],
        "lab_pre24h",
    )

    outputs: dict[int, pd.DataFrame] = {}
    predictor_lists: dict[int, list[str]] = {}
    summary_rows: list[dict[str, object]] = []
    missingness_rows: list[dict[str, object]] = []
    excluded = leakage_exclusions(cohort.columns)

    for landmark in [0, 6, 24]:
        dataset, excluded_early, predictors = build_landmark_dataset(
            cohort, preindex_labs, labs_aligned, vitals_aligned, landmark
        )
        outputs[landmark] = dataset
        predictor_lists[landmark] = predictors
        event = bool_mask(dataset["outcome_aki_after_landmark_to_7d"])
        summary_rows.append(
            {
                "landmark_hours": landmark,
                "sample_size": len(dataset),
                "event_n": int(event.sum()),
                "event_incidence_percent": round(float(event.mean() * 100), 2),
                "excluded_aki_at_or_before_landmark_n": excluded_early,
                "predictor_count": len(predictors),
                "excluded_leakage_variable_count": len(excluded),
                "excluded_leakage_variables": " | ".join(excluded),
            }
        )
        for column in predictors:
            missingness_rows.append(
                {
                    "landmark_hours": landmark,
                    "predictor": column,
                    "n_missing": int(dataset[column].isna().sum()),
                    "missing_percent": round(float(dataset[column].isna().mean() * 100), 3),
                    "n_observed": int(dataset[column].notna().sum()),
                }
            )
        dataset.to_csv(OUTPUT_DIR / f"dataset_v4_{landmark}h.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "audit_v4_dynamic_dataset_summary.csv", index=False)
    pd.DataFrame(missingness_rows).to_csv(
        OUTPUT_DIR / "audit_v4_predictor_missingness.csv", index=False
    )
    write_readme(summary, excluded, predictor_lists)
    print("\n" + summary.drop(columns=["excluded_leakage_variables"]).to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
