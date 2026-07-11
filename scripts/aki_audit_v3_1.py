"""Audit v3 AKI outcomes and run the pre-index baseline sensitivity analysis.

Reads only the two finalized v3 cohort CSV files. No raw MIMIC tables and no
machine-learning procedures are used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "outputs" / "finalized_v3"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "finalized_v3_1"

STRICT_INPUT = INPUT_DIR / "cohort_v3_strict_primary_aki_final.csv"
BROAD_INPUT = INPUT_DIR / "cohort_v3_broad_sensitivity_aki_final.csv"

SURGERY_FLAGS = [
    "cardiac_surgery",
    "non_cardiac_surgery",
    "vascular_surgery",
    "general_gi_hepatobiliary_surgery",
    "orthopedic_major_surgery",
    "neurosurgery",
    "thoracic_respiratory_surgery",
]


def require_columns(data: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def load_cohort(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path, low_memory=False)
    require_columns(
        data,
        [
            "subject_id", "hadm_id", "stay_id", "intime", "baseline_scr_time",
            "baseline_scr_source", "aki_final", "aki_stage_final",
            "aki_onset_hours_final", "incident_aki_evaluable",
            "aki_provisional", "aki_label_discordant",
        ],
        label,
    )
    for column in ["intime", "baseline_scr_time", "aki_onset_time_final"]:
        data[column] = pd.to_datetime(data[column], errors="coerce")
    if data["hadm_id"].duplicated().any():
        raise ValueError(f"{label} is not unique by hadm_id")
    return data


def add_audit_variables(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["baseline_to_icu_hours"] = (
        result["intime"] - result["baseline_scr_time"]
    ).dt.total_seconds() / 3600
    result["aki_onset_window_v3_1"] = pd.cut(
        pd.to_numeric(result["aki_onset_hours_final"], errors="coerce"),
        bins=[-np.inf, 24, 48, 72, 168],
        labels=["0-24h", ">24-48h", ">48-72h", ">72h-7d"],
        right=True,
    ).astype("string")
    return result


def summarize_view(view_name: str, data: pd.DataFrame) -> dict[str, object]:
    aki = bool_mask(data["aki_final"])
    discordant = bool_mask(data["aki_label_discordant"])
    onset = pd.to_numeric(data.loc[aki, "aki_onset_hours_final"], errors="coerce")
    baseline_delta = pd.to_numeric(data["baseline_to_icu_hours"], errors="coerce")
    row: dict[str, object] = {
        "analysis_view": view_name,
        "n": len(data),
        "unique_subjects_n": data["subject_id"].nunique(),
        "unique_hadm_id_n": data["hadm_id"].nunique(),
        "aki_n": int(aki.sum()),
        "aki_incidence_percent": round(float(aki.mean() * 100), 2),
        "provisional_aki_n": int((pd.to_numeric(data["aki_provisional"], errors="coerce") == 1).sum()),
        "discordant_n": int(discordant.sum()),
        "discordant_percent": round(float(discordant.mean() * 100), 2),
        "provisional_0_final_1_n": int(
            ((pd.to_numeric(data["aki_provisional"], errors="coerce") == 0) & aki).sum()
        ),
        "provisional_1_final_0_n": int(
            ((pd.to_numeric(data["aki_provisional"], errors="coerce") == 1) & ~aki).sum()
        ),
        "baseline_to_icu_hours_median": round(float(baseline_delta.median()), 2),
        "baseline_to_icu_hours_q1": round(float(baseline_delta.quantile(0.25)), 2),
        "baseline_to_icu_hours_q3": round(float(baseline_delta.quantile(0.75)), 2),
        "baseline_after_icu_n": int((baseline_delta < 0).sum()),
        "aki_onset_hours_median": round(float(onset.median()), 2) if onset.notna().any() else np.nan,
        "aki_onset_hours_q1": round(float(onset.quantile(0.25)), 2) if onset.notna().any() else np.nan,
        "aki_onset_hours_q3": round(float(onset.quantile(0.75)), 2) if onset.notna().any() else np.nan,
    }
    stages = pd.to_numeric(data.loc[aki, "aki_stage_final"], errors="coerce")
    for stage in [1, 2, 3]:
        count = int((stages == stage).sum())
        row[f"aki_stage_{stage}_n"] = count
        row[f"aki_stage_{stage}_percent_among_aki"] = round(
            count / int(aki.sum()) * 100, 2
        ) if aki.any() else np.nan
    for window in ["0-24h", ">24-48h", ">48-72h", ">72h-7d"]:
        count = int(data.loc[aki, "aki_onset_window_v3_1"].eq(window).sum())
        safe_name = (
            window.replace(">", "gt_").replace("-", "_to_").replace("h", "h").replace("7d", "7d")
        )
        row[f"aki_onset_{safe_name}_n"] = count
        row[f"aki_onset_{safe_name}_percent"] = round(
            count / int(aki.sum()) * 100, 2
        ) if aki.any() else np.nan
    return row


def baseline_sensitivity_rows(view_name: str, data: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source = data["baseline_scr_source"].fillna("<missing>")
    for category, n in source.value_counts(dropna=False).items():
        rows.append(
            {
                "analysis_view": view_name,
                "dimension": "baseline_scr_source",
                "category": str(category),
                "n": int(n),
                "percent": round(float(n / len(data) * 100), 2),
                "median_hours": np.nan,
                "q1_hours": np.nan,
                "q3_hours": np.nan,
            }
        )

    delta = pd.to_numeric(data["baseline_to_icu_hours"], errors="coerce")
    delta_categories = pd.Series("<missing>", index=data.index, dtype="string")
    delta_categories.loc[delta < 0] = "baseline_after_icu"
    delta_categories.loc[delta.between(0, 24, inclusive="both")] = "0-24h_before_icu"
    delta_categories.loc[(delta > 24) & (delta <= 48)] = ">24-48h_before_icu"
    delta_categories.loc[(delta > 48) & (delta <= 168)] = ">48h-7d_before_icu"
    delta_categories.loc[delta > 168] = ">7d_before_icu"
    for category, n in delta_categories.value_counts().items():
        rows.append(
            {
                "analysis_view": view_name,
                "dimension": "baseline_to_icu_time_window",
                "category": str(category),
                "n": int(n),
                "percent": round(float(n / len(data) * 100), 2),
                "median_hours": np.nan,
                "q1_hours": np.nan,
                "q3_hours": np.nan,
            }
        )
    rows.append(
        {
            "analysis_view": view_name,
            "dimension": "baseline_to_icu_hours_summary",
            "category": "all_observed",
            "n": int(delta.notna().sum()),
            "percent": round(float(delta.notna().mean() * 100), 2),
            "median_hours": round(float(delta.median()), 2),
            "q1_hours": round(float(delta.quantile(0.25)), 2),
            "q3_hours": round(float(delta.quantile(0.75)), 2),
        }
    )
    return rows


def surgery_subgroup_rows(data: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for flag in SURGERY_FLAGS:
        require_columns(data, [flag], "strict main evaluable cohort")
        known = data[flag].notna()
        subgroup = bool_mask(data[flag])
        selected = data.loc[subgroup]
        aki = bool_mask(selected["aki_final"])
        stages = pd.to_numeric(selected.loc[aki, "aki_stage_final"], errors="coerce")
        rows.append(
            {
                "surgery_subgroup": flag,
                "strict_evaluable_total_n": len(data),
                "flag_known_n": int(known.sum()),
                "subgroup_n": len(selected),
                "subgroup_percent": round(len(selected) / len(data) * 100, 2),
                "aki_n": int(aki.sum()),
                "aki_incidence_percent": round(float(aki.mean() * 100), 2) if len(selected) else np.nan,
                "aki_stage_1_n": int((stages == 1).sum()),
                "aki_stage_2_n": int((stages == 2).sum()),
                "aki_stage_3_n": int((stages == 3).sum()),
            }
        )
    return rows


def write_readme(summary: pd.DataFrame, subgroup: pd.DataFrame) -> None:
    s = summary.set_index("analysis_view")
    main = s.loc["strict_main_evaluable"]
    sensitivity = s.loc["strict_pre_index_baseline_only"]
    broad = s.loc["broad_main_evaluable"]
    content = f"""# v3.1 AKI outcome audit and baseline sensitivity

## Scope

- Main denominator: `incident_aki_evaluable == True`.
- Strict main evaluable cohort: all evaluable rows from the strict v3 cohort.
- Strict pre-index baseline sensitivity cohort: evaluable strict rows with `baseline_scr_source == lowest_scr_7d_pre_icu`.
- Broad evaluable cohort is summarized for context but is not exported as a new patient-level file.
- No raw MIMIC tables were read and no machine learning was performed.

## Main comparison

| Analysis view | N | AKI n | AKI incidence | Discordance | Baseline-to-ICU median | AKI onset median |
|---|---:|---:|---:|---:|---:|---:|
| Strict main evaluable | {int(main['n']):,} | {int(main['aki_n']):,} | {main['aki_incidence_percent']:.2f}% | {main['discordant_percent']:.2f}% | {main['baseline_to_icu_hours_median']:.2f} h | {main['aki_onset_hours_median']:.2f} h |
| Strict pre-index baseline only | {int(sensitivity['n']):,} | {int(sensitivity['aki_n']):,} | {sensitivity['aki_incidence_percent']:.2f}% | {sensitivity['discordant_percent']:.2f}% | {sensitivity['baseline_to_icu_hours_median']:.2f} h | {sensitivity['aki_onset_hours_median']:.2f} h |
| Broad main evaluable | {int(broad['n']):,} | {int(broad['aki_n']):,} | {broad['aki_incidence_percent']:.2f}% | {broad['discordant_percent']:.2f}% | {broad['baseline_to_icu_hours_median']:.2f} h | {broad['aki_onset_hours_median']:.2f} h |

Positive baseline-to-ICU hours mean the baseline preceded ICU admission; negative values mean the admission fallback was charted after ICU admission. The pre-index sensitivity cohort contains no post-index baselines by definition.

## Stage interpretation

Stage percentages in `audit_v3_1_aki_summary.csv` use final AKI cases—not the full cohort—as their denominator. AKI incidence and provisional/final discordance use all rows in the corresponding evaluable view.

## Surgical subgroups

Subgroups are overlapping where multiple surgical-category flags are true. `non_cardiac_surgery` is based on the v2/v3 category definition and is not derived again in this audit. Rates in `audit_v3_1_surgery_subgroup_aki.csv` use evaluable subgroup members as denominators.

## Outputs

- `cohort_v3_1_strict_main_evaluable.csv`
- `cohort_v3_1_strict_pre_index_baseline_only.csv`
- `audit_v3_1_aki_summary.csv`
- `audit_v3_1_baseline_sensitivity.csv`
- `audit_v3_1_surgery_subgroup_aki.csv`

No machine learning was performed.
"""
    (OUTPUT_DIR / "audit_v3_1_readme.md").write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strict = add_audit_variables(load_cohort(STRICT_INPUT, "strict v3 cohort"))
    broad = add_audit_variables(load_cohort(BROAD_INPUT, "broad v3 cohort"))

    strict_main = strict.loc[bool_mask(strict["incident_aki_evaluable"])].copy()
    strict_pre_index = strict_main.loc[
        strict_main["baseline_scr_source"].eq("lowest_scr_7d_pre_icu")
    ].copy()
    broad_main = broad.loc[bool_mask(broad["incident_aki_evaluable"])].copy()

    strict_main.insert(0, "v3_1_analysis_view", "strict_main_evaluable")
    strict_pre_index.insert(0, "v3_1_analysis_view", "strict_pre_index_baseline_only")
    strict_main.to_csv(OUTPUT_DIR / "cohort_v3_1_strict_main_evaluable.csv", index=False)
    strict_pre_index.to_csv(
        OUTPUT_DIR / "cohort_v3_1_strict_pre_index_baseline_only.csv", index=False
    )

    summary = pd.DataFrame(
        [
            summarize_view("strict_main_evaluable", strict_main),
            summarize_view("strict_pre_index_baseline_only", strict_pre_index),
            summarize_view("broad_main_evaluable", broad_main),
        ]
    )
    summary.to_csv(OUTPUT_DIR / "audit_v3_1_aki_summary.csv", index=False)

    baseline = pd.DataFrame(
        baseline_sensitivity_rows("strict_main_evaluable", strict_main)
        + baseline_sensitivity_rows("strict_pre_index_baseline_only", strict_pre_index)
        + baseline_sensitivity_rows("broad_main_evaluable", broad_main)
    )
    baseline.to_csv(OUTPUT_DIR / "audit_v3_1_baseline_sensitivity.csv", index=False)

    subgroup = pd.DataFrame(surgery_subgroup_rows(strict_main))
    subgroup.to_csv(OUTPUT_DIR / "audit_v3_1_surgery_subgroup_aki.csv", index=False)

    write_readme(summary, subgroup)
    print(summary.to_string(index=False))
    print("\nStrict evaluable surgery subgroups:")
    print(subgroup.to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

