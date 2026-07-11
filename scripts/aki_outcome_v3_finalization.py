"""Finalize postoperative incident AKI outcomes using KDIGO serum creatinine.

Inputs are the two v2 finalized cohorts plus MIMIC-IV labevents serum
creatinine (itemid 50912). Urine output and machine learning are intentionally
out of scope.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIMIC_ROOT = PROJECT_ROOT.parent
INPUT_DIR = PROJECT_ROOT / "outputs" / "finalized_v2"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "finalized_v3"
CACHE_DIR = PROJECT_ROOT / "outputs" / "cache_v3"

STRICT_INPUT = INPUT_DIR / "cohort_v2_strict_primary.csv"
BROAD_INPUT = INPUT_DIR / "cohort_v2_broad_sensitivity.csv"
LABEVENTS = MIMIC_ROOT / "hosp" / "labevents.csv"

CREATININE_ITEMID = 50912
CHUNK_SIZE = 2_000_000
LOOKBACK_7D = pd.Timedelta(days=7)
ROLLING_48H = pd.Timedelta(hours=48)
FOLLOWUP_7D = pd.Timedelta(days=7)


def require_columns(data: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_cohort(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    data = pd.read_csv(path, low_memory=False)
    require_columns(
        data,
        ["subject_id", "hadm_id", "stay_id", "intime", "admittime", "aki"],
        label,
    )
    for column in ["intime", "admittime", "outtime", "dischtime", "index_surgery_date"]:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce")
    if data["hadm_id"].duplicated().any():
        raise ValueError(f"{label} is not unique by hadm_id")
    return data


def extract_creatinine_labs(subject_ids: set[int]) -> pd.DataFrame:
    """Retain SCr for cohort subjects, including rows with a missing hadm_id."""
    if not LABEVENTS.exists():
        raise FileNotFoundError(LABEVENTS)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "creatinine_subject_labs.csv.gz"
    if cache_path.exists():
        print(f"Loading cached subject-level creatinine rows: {cache_path}", flush=True)
        cached = pd.read_csv(cache_path, low_memory=False, parse_dates=["charttime"])
        cached_subjects = set(cached["subject_id"].astype(int).unique())
        if subject_ids.issubset(cached_subjects):
            return cached.loc[cached["subject_id"].isin(subject_ids)].copy()
        print("Cache does not cover every cohort subject; rebuilding it.", flush=True)
    retained: list[pd.DataFrame] = []
    usecols = ["labevent_id", "subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"]
    for chunk_number, chunk in enumerate(
        pd.read_csv(LABEVENTS, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False), start=1
    ):
        selected = chunk.loc[
            chunk["itemid"].eq(CREATININE_ITEMID)
            & chunk["subject_id"].isin(subject_ids)
            & chunk["valuenum"].notna()
        ].copy()
        if not selected.empty:
            selected["charttime"] = pd.to_datetime(selected["charttime"], errors="coerce")
            selected = selected.loc[
                selected["charttime"].notna()
                & selected["valuenum"].gt(0)
                & selected["valuenum"].le(100)
            ]
            retained.append(selected)
        if chunk_number % 10 == 0:
            print(f"Scanned {chunk_number * CHUNK_SIZE:,} labevents rows...", flush=True)
    if not retained:
        return pd.DataFrame(columns=usecols)
    labs = pd.concat(retained, ignore_index=True)
    labs["valuenum"] = pd.to_numeric(labs["valuenum"], errors="coerce")
    labs = labs.sort_values(["subject_id", "charttime", "labevent_id"])
    labs.to_csv(cache_path, index=False, compression="gzip")
    return labs


def has_rolling_absolute_rise(times: pd.Series, values: pd.Series) -> tuple[bool, pd.Timestamp | pd.NaT]:
    """Return whether a value rose >=0.3 from any prior value in the preceding 48 h."""
    window: deque[tuple[pd.Timestamp, float]] = deque()
    first_event = pd.NaT
    for time, value in zip(times, values):
        while window and time - window[0][0] > ROLLING_48H:
            window.popleft()
        if window and value - min(item[1] for item in window) >= 0.3 - 1e-12:
            first_event = time
            return True, first_event
        window.append((time, float(value)))
    return False, first_event


def derive_one_outcome(row: pd.Series, labs: pd.DataFrame) -> dict[str, object]:
    index_time = row["intime"]
    admission_time = row["admittime"]
    followup_end = index_time + FOLLOWUP_7D
    patient_labs = labs.sort_values(["charttime", "labevent_id"]).copy()

    pre_context_start = min(admission_time, index_time - LOOKBACK_7D)
    pre_index = patient_labs.loc[
        patient_labs["charttime"].ge(pre_context_start)
        & patient_labs["charttime"].le(index_time)
    ].copy()
    preferred = pre_index.loc[
        pre_index["charttime"].ge(index_time - LOOKBACK_7D)
        & pre_index["charttime"].lt(index_time)
    ]
    if not preferred.empty:
        baseline_row = preferred.loc[preferred["valuenum"].idxmin()]
        baseline_source = "lowest_scr_7d_pre_icu"
    else:
        admission_candidates = patient_labs.loc[
            patient_labs["charttime"].ge(admission_time)
            & patient_labs["charttime"].le(admission_time + pd.Timedelta(hours=24))
        ]
        if not admission_candidates.empty:
            baseline_row = admission_candidates.sort_values(
                ["charttime", "labevent_id"]
            ).iloc[0]
            baseline_source = "admission_scr_first_24h_fallback"
        else:
            baseline_row = None
            baseline_source = "missing"

    baseline_scr = float(baseline_row["valuenum"]) if baseline_row is not None else np.nan
    baseline_time = baseline_row["charttime"] if baseline_row is not None else pd.NaT

    pre_abs_aki, pre_abs_time = has_rolling_absolute_rise(
        pre_index["charttime"], pre_index["valuenum"]
    )
    pre_ratio_aki = False
    pre_ratio_time = pd.NaT
    if not np.isnan(baseline_scr) and not pre_index.empty:
        ratio_hits = pre_index.loc[pre_index["valuenum"].ge(1.5 * baseline_scr)]
        if not ratio_hits.empty:
            pre_ratio_aki = True
            pre_ratio_time = ratio_hits.iloc[0]["charttime"]
    preexisting_aki = bool(pre_abs_aki or pre_ratio_aki)
    pre_times = [time for time in [pre_abs_time, pre_ratio_time] if pd.notna(time)]
    preexisting_aki_time = min(pre_times) if pre_times else pd.NaT

    post = patient_labs.loc[
        patient_labs["charttime"].gt(index_time)
        & patient_labs["charttime"].le(followup_end)
    ].copy()
    n_post = len(post)

    result: dict[str, object] = {
        "baseline_scr_final": baseline_scr,
        "baseline_scr_time": baseline_time,
        "baseline_scr_source": baseline_source,
        "preexisting_aki_at_or_before_index": preexisting_aki,
        "preexisting_aki_first_time": preexisting_aki_time,
        "post_index_scr_n_7d": n_post,
        "post_index_first_scr_time": post["charttime"].min() if n_post else pd.NaT,
        "post_index_peak_scr_7d": post["valuenum"].max() if n_post else np.nan,
        "post_index_peak_scr_time": (
            post.loc[post["valuenum"].idxmax(), "charttime"] if n_post else pd.NaT
        ),
        "aki_final": pd.NA,
        "aki_stage_final": pd.NA,
        "aki_onset_time_final": pd.NaT,
        "aki_onset_hours_final": np.nan,
        "aki_criterion_0_3_within_48h": pd.NA,
        "aki_criterion_1_5x_within_7d": pd.NA,
        "peak_scr_ratio_to_baseline": np.nan,
        "incident_aki_evaluable": False,
        "incident_aki_ineligibility_reason": "",
    }

    if np.isnan(baseline_scr):
        result["incident_aki_ineligibility_reason"] = "baseline_scr_missing"
        return result
    if preexisting_aki:
        result["incident_aki_ineligibility_reason"] = "aki_present_before_or_at_index"
        return result
    if post.empty:
        result["incident_aki_ineligibility_reason"] = "no_post_index_scr_within_7d"
        return result

    result["incident_aki_evaluable"] = True
    result["incident_aki_ineligibility_reason"] = "eligible"

    # Criterion A can use a prior creatinine from up to 48 h before a post-index result.
    rolling_source = patient_labs.loc[
        patient_labs["charttime"].ge(index_time - ROLLING_48H)
        & patient_labs["charttime"].le(followup_end)
    ].copy()
    rolling_window: deque[tuple[pd.Timestamp, float]] = deque()
    absolute_event_times: list[pd.Timestamp] = []
    for _, lab in rolling_source.iterrows():
        time = lab["charttime"]
        value = float(lab["valuenum"])
        while rolling_window and time - rolling_window[0][0] > ROLLING_48H:
            rolling_window.popleft()
        if time > index_time and rolling_window:
            if value - min(item[1] for item in rolling_window) >= 0.3 - 1e-12:
                absolute_event_times.append(time)
        rolling_window.append((time, value))

    ratio_hits = post.loc[post["valuenum"].ge(1.5 * baseline_scr)]
    absolute_hit = bool(absolute_event_times)
    ratio_hit = not ratio_hits.empty
    event_times = absolute_event_times + ratio_hits["charttime"].tolist()
    aki_final = bool(absolute_hit or ratio_hit)
    onset_time = min(event_times) if event_times else pd.NaT
    peak_scr = float(post["valuenum"].max())
    peak_ratio = peak_scr / baseline_scr

    stage = 0
    if aki_final:
        if peak_ratio >= 3.0 or peak_scr >= 4.0:
            stage = 3
        elif peak_ratio >= 2.0:
            stage = 2
        else:
            stage = 1

    result.update(
        {
            "aki_final": aki_final,
            "aki_stage_final": stage,
            "aki_onset_time_final": onset_time,
            "aki_onset_hours_final": (
                (onset_time - index_time).total_seconds() / 3600 if pd.notna(onset_time) else np.nan
            ),
            "aki_criterion_0_3_within_48h": absolute_hit,
            "aki_criterion_1_5x_within_7d": ratio_hit,
            "peak_scr_ratio_to_baseline": peak_ratio,
        }
    )
    return result


def derive_outcomes(index_rows: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    labs_by_subject = {
        subject_id: group for subject_id, group in labs.groupby("subject_id", sort=False)
    }
    empty = labs.iloc[0:0].copy()
    records: list[dict[str, object]] = []
    for number, (_, row) in enumerate(index_rows.iterrows(), start=1):
        records.append(
            derive_one_outcome(row, labs_by_subject.get(int(row["subject_id"]), empty))
        )
        if number % 5_000 == 0:
            print(f"Derived AKI outcomes for {number:,}/{len(index_rows):,} admissions...", flush=True)
    outcome = pd.DataFrame(records, index=index_rows.index)
    outcome.insert(0, "stay_id", index_rows["stay_id"].values)
    return outcome


def attach_outcomes(cohort: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    renamed = cohort.rename(
        columns={
            "aki": "aki_provisional",
            "aki_stage": "aki_stage_provisional",
            "aki_onset_time": "aki_onset_time_provisional",
            "aki_onset_days": "aki_onset_days_provisional",
            "baseline_scr": "baseline_scr_provisional",
        }
    )
    final = renamed.merge(outcomes, on="stay_id", how="left", validate="one_to_one")
    comparable = final["aki_final"].notna() & final["aki_provisional"].notna()
    final["aki_label_comparable"] = comparable
    final["aki_label_discordant"] = pd.Series(pd.NA, index=final.index, dtype="boolean")
    final.loc[comparable, "aki_label_discordant"] = (
        final.loc[comparable, "aki_provisional"].astype(int)
        != final.loc[comparable, "aki_final"].astype(int)
    )
    return final


def binary_count_rate(data: pd.DataFrame, column: str) -> tuple[int, float]:
    observed = pd.to_numeric(data[column], errors="coerce").dropna()
    if observed.empty:
        return 0, np.nan
    return int((observed == 1).sum()), round(float((observed == 1).mean() * 100), 2)


def outcome_summary(cohort_name: str, data: pd.DataFrame) -> dict[str, object]:
    evaluable = data["incident_aki_evaluable"].fillna(False).astype(bool)
    comparable = data["aki_label_comparable"].fillna(False).astype(bool)
    final_aki_n, final_aki_rate = binary_count_rate(data.loc[evaluable], "aki_final")
    provisional_n, provisional_rate = binary_count_rate(data, "aki_provisional")
    comparison = data.loc[comparable]
    old = pd.to_numeric(comparison["aki_provisional"], errors="coerce")
    new = pd.to_numeric(comparison["aki_final"], errors="coerce")
    return {
        "cohort": cohort_name,
        "total_n": len(data),
        "incident_aki_evaluable_n": int(evaluable.sum()),
        "preexisting_aki_n": int(data["preexisting_aki_at_or_before_index"].fillna(False).sum()),
        "baseline_missing_n": int(data["baseline_scr_source"].eq("missing").sum()),
        "no_post_index_scr_n": int(data["incident_aki_ineligibility_reason"].eq("no_post_index_scr_within_7d").sum()),
        "aki_final_n": final_aki_n,
        "aki_final_incidence_percent_among_evaluable": final_aki_rate,
        "aki_provisional_n": provisional_n,
        "aki_provisional_percent": provisional_rate,
        "labels_comparable_n": int(comparable.sum()),
        "labels_discordant_n": int((old != new).sum()),
        "labels_discordant_percent": round(float((old != new).mean() * 100), 2) if len(comparison) else np.nan,
        "provisional_0_final_1_n": int(((old == 0) & (new == 1)).sum()),
        "provisional_1_final_0_n": int(((old == 1) & (new == 0)).sum()),
        "both_criteria_n": int(
            (
                data["aki_criterion_0_3_within_48h"].fillna(False).astype(bool)
                & data["aki_criterion_1_5x_within_7d"].fillna(False).astype(bool)
            ).sum()
        ),
        "absolute_0_3_criterion_n": int(data["aki_criterion_0_3_within_48h"].fillna(False).sum()),
        "ratio_1_5_criterion_n": int(data["aki_criterion_1_5x_within_7d"].fillna(False).sum()),
    }


def baseline_summary_rows(cohort_name: str, data: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source_text = data["baseline_scr_source"].fillna("missing")
    for source in ["all", *source_text.value_counts().index.tolist()]:
        subset = data if source == "all" else data.loc[source_text.eq(source)]
        values = pd.to_numeric(subset["baseline_scr_final"], errors="coerce")
        rows.append(
            {
                "cohort": cohort_name,
                "baseline_source": source,
                "n": len(subset),
                "percent_of_cohort": round(len(subset) / len(data) * 100, 2),
                "baseline_scr_observed_n": int(values.notna().sum()),
                "baseline_scr_mean": round(float(values.mean()), 3) if values.notna().any() else np.nan,
                "baseline_scr_sd": round(float(values.std()), 3) if values.notna().any() else np.nan,
                "baseline_scr_median": round(float(values.median()), 3) if values.notna().any() else np.nan,
                "baseline_scr_q1": round(float(values.quantile(0.25)), 3) if values.notna().any() else np.nan,
                "baseline_scr_q3": round(float(values.quantile(0.75)), 3) if values.notna().any() else np.nan,
                "baseline_scr_min": round(float(values.min()), 3) if values.notna().any() else np.nan,
                "baseline_scr_max": round(float(values.max()), 3) if values.notna().any() else np.nan,
            }
        )
    return rows


def timing_summary_rows(cohort_name: str, data: pd.DataFrame) -> list[dict[str, object]]:
    evaluable = data.loc[data["incident_aki_evaluable"].fillna(False).astype(bool)].copy()
    aki = evaluable.loc[evaluable["aki_final"].fillna(False).astype(bool)].copy()
    rows: list[dict[str, object]] = []

    for stage, count in aki["aki_stage_final"].value_counts(dropna=False).sort_index().items():
        rows.append(
            {
                "cohort": cohort_name,
                "dimension": "aki_stage_final",
                "category": str(stage),
                "n": int(count),
                "percent_among_final_aki": round(count / len(aki) * 100, 2) if len(aki) else np.nan,
                "onset_hours_median": np.nan,
                "onset_hours_q1": np.nan,
                "onset_hours_q3": np.nan,
            }
        )

    bins = [-np.inf, 24, 48, 72, 168]
    labels = ["0-24h", ">24-48h", ">48-72h", ">72h-7d"]
    timing_bins = pd.cut(aki["aki_onset_hours_final"], bins=bins, labels=labels, right=True)
    for label in labels:
        count = int(timing_bins.eq(label).sum())
        rows.append(
            {
                "cohort": cohort_name,
                "dimension": "aki_onset_window",
                "category": label,
                "n": count,
                "percent_among_final_aki": round(count / len(aki) * 100, 2) if len(aki) else np.nan,
                "onset_hours_median": np.nan,
                "onset_hours_q1": np.nan,
                "onset_hours_q3": np.nan,
            }
        )

    onset = pd.to_numeric(aki["aki_onset_hours_final"], errors="coerce")
    rows.append(
        {
            "cohort": cohort_name,
            "dimension": "aki_onset_hours_summary",
            "category": "all_final_aki",
            "n": int(onset.notna().sum()),
            "percent_among_final_aki": 100.0 if len(aki) else np.nan,
            "onset_hours_median": round(float(onset.median()), 2) if onset.notna().any() else np.nan,
            "onset_hours_q1": round(float(onset.quantile(0.25)), 2) if onset.notna().any() else np.nan,
            "onset_hours_q3": round(float(onset.quantile(0.75)), 2) if onset.notna().any() else np.nan,
        }
    )
    return rows


def write_readme(strict: pd.DataFrame, broad: pd.DataFrame, summary: pd.DataFrame) -> None:
    s = summary.set_index("cohort")
    content = f"""# AKI outcome finalization v3

## Scope

- Primary cohort: `cohort_v2_strict_primary.csv`.
- Sensitivity cohort: `cohort_v2_broad_sensitivity.csv`.
- Analysis unit: first ICU stay per hospital admission.
- Index time: ICU `intime`; `index_surgery_date` is retained for audit where available.
- Serum creatinine only (`labevents.itemid = 50912`); urine output was not used.
- No machine learning was performed.

## Baseline creatinine

1. Lowest valid serum creatinine strictly before ICU admission within the preceding 7 days.
2. If unavailable, the earliest serum creatinine within 24 hours after hospital `admittime`. For direct-to-ICU admissions this sample may be charted shortly after ICU `intime`; its actual timestamp is retained and all subsequent comparisons remain chronological.
3. If neither exists, baseline is missing and incident AKI is not evaluated.

`baseline_scr_source` records `lowest_scr_7d_pre_icu`, `admission_scr_first_24h_fallback`, or `missing`.

## Incident AKI

The outcome window is strictly after ICU `intime` through 7 days. AKI is present when either:

- SCr rises by at least 0.3 mg/dL from a prior result within 48 hours; or
- SCr reaches at least 1.5 times the final baseline within 7 days.

Patients meeting a rolling 0.3-mg/dL or 1.5-times-baseline criterion at/before index are flagged as prevalent AKI. They remain in the CSV but `aki_final` is missing and they are excluded from the incident-AKI denominator. The same applies to missing baseline or no post-index SCr.

Stage is assigned from the 7-day post-index peak among incident AKI cases: Stage 1 (1.5–<2.0 times baseline or 0.3-mg/dL criterion), Stage 2 (2.0–<3.0), Stage 3 (≥3.0 or peak SCr ≥4.0 mg/dL).

## Results

| Cohort | Total | Evaluable | Final AKI | Incidence | Prevalent/index AKI | Discordant labels |
|---|---:|---:|---:|---:|---:|---:|
| Strict primary | {int(s.loc['strict_primary','total_n']):,} | {int(s.loc['strict_primary','incident_aki_evaluable_n']):,} | {int(s.loc['strict_primary','aki_final_n']):,} | {s.loc['strict_primary','aki_final_incidence_percent_among_evaluable']:.2f}% | {int(s.loc['strict_primary','preexisting_aki_n']):,} | {int(s.loc['strict_primary','labels_discordant_n']):,} |
| Broad sensitivity | {int(s.loc['broad_sensitivity','total_n']):,} | {int(s.loc['broad_sensitivity','incident_aki_evaluable_n']):,} | {int(s.loc['broad_sensitivity','aki_final_n']):,} | {s.loc['broad_sensitivity','aki_final_incidence_percent_among_evaluable']:.2f}% | {int(s.loc['broad_sensitivity','preexisting_aki_n']):,} | {int(s.loc['broad_sensitivity','labels_discordant_n']):,} |

## Provisional-label comparison

Discordance is calculated only where `aki_final` is evaluable. The two directional counts are reported separately in `audit_v3_aki_outcome_summary.csv` as provisional 0→final 1 and provisional 1→final 0.

## Outputs

- `cohort_v3_strict_primary_aki_final.csv`
- `cohort_v3_broad_sensitivity_aki_final.csv`
- `audit_v3_aki_outcome_summary.csv`
- `audit_v3_baseline_creatinine_summary.csv`
- `audit_v3_aki_timing_summary.csv`
"""
    (OUTPUT_DIR / "audit_v3_readme.md").write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strict = load_cohort(STRICT_INPUT, "strict v2 cohort")
    broad = load_cohort(BROAD_INPUT, "broad v2 cohort")

    combined_index = (
        pd.concat(
            [
                strict[["subject_id", "hadm_id", "stay_id", "intime", "admittime"]],
                broad[["subject_id", "hadm_id", "stay_id", "intime", "admittime"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates("stay_id")
        .copy()
    )
    if combined_index["hadm_id"].duplicated().any():
        raise ValueError("Combined v2 index contains multiple stay_ids for a hadm_id")

    labs = extract_creatinine_labs(set(combined_index["subject_id"].astype(int)))
    print(f"Retained {len(labs):,} serum-creatinine rows for v3 cohorts.", flush=True)
    outcomes = derive_outcomes(combined_index, labs)

    strict_final = attach_outcomes(strict, outcomes)
    broad_final = attach_outcomes(broad, outcomes)

    strict_final.to_csv(OUTPUT_DIR / "cohort_v3_strict_primary_aki_final.csv", index=False)
    broad_final.to_csv(OUTPUT_DIR / "cohort_v3_broad_sensitivity_aki_final.csv", index=False)

    summary = pd.DataFrame(
        [
            outcome_summary("strict_primary", strict_final),
            outcome_summary("broad_sensitivity", broad_final),
        ]
    )
    summary.to_csv(OUTPUT_DIR / "audit_v3_aki_outcome_summary.csv", index=False)

    baseline_summary = pd.DataFrame(
        baseline_summary_rows("strict_primary", strict_final)
        + baseline_summary_rows("broad_sensitivity", broad_final)
    )
    baseline_summary.to_csv(OUTPUT_DIR / "audit_v3_baseline_creatinine_summary.csv", index=False)

    timing_summary = pd.DataFrame(
        timing_summary_rows("strict_primary", strict_final)
        + timing_summary_rows("broad_sensitivity", broad_final)
    )
    timing_summary.to_csv(OUTPUT_DIR / "audit_v3_aki_timing_summary.csv", index=False)

    write_readme(strict_final, broad_final, summary)
    print("\n" + summary.to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
