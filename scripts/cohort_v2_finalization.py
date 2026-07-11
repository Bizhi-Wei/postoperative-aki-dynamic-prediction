"""Finalize strict and broad postoperative-AKI cohorts from screening outputs only.

This script deliberately does not read any raw MIMIC-IV table. Its only cohort
inputs are:
  - cohort_version_a_all_stays_audit.csv
  - cohort_version_b_all_stays_audit.csv

Analysis unit: the first ICU stay within each hospital admission (hadm_id).
No machine-learning analysis is performed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "outputs" / "screening"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "finalized_v2"

INPUT_A = INPUT_DIR / "cohort_version_a_all_stays_audit.csv"
INPUT_B = INPUT_DIR / "cohort_version_b_all_stays_audit.csv"

ANALYSIS_UNIT = "first ICU stay per hospital admission (hadm_id)"

CATEGORY_TO_FLAG = {
    "cardiac": "cardiac_surgery",
    "vascular": "vascular_surgery",
    "general_gi_hepatobiliary": "general_gi_hepatobiliary_surgery",
    "orthopedic_major": "orthopedic_major_surgery",
    "neurosurgery": "neurosurgery",
    "thoracic_respiratory": "thoracic_respiratory_surgery",
}

OBSTETRIC_PATTERN = re.compile(
    r"obstetric|cesarean|caesarean|delivery|abortion|fetal|placenta|puerper|episiotomy",
    flags=re.IGNORECASE,
)

DEATH_COLUMNS = ["icu_death", "hosp_death", "death_90d", "death_365d"]


def require_columns(data: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_input(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} input: {path}")
    data = pd.read_csv(path, low_memory=False)
    require_columns(data, ["subject_id", "hadm_id", "stay_id", "intime", "aki", "los"], label)
    data["intime"] = pd.to_datetime(data["intime"], errors="coerce")
    if data["hadm_id"].isna().any():
        raise ValueError(f"{label} contains missing hadm_id; analysis-unit selection is unsafe")
    # The forbidden legacy label is removed before any downstream processing.
    data = data.drop(columns=["primary_surgery"], errors="ignore")
    return data


def remove_obstetric_cases(strict_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    searchable_columns = [
        column
        for column in ["surgery_categories", "qualifying_procedure_titles"]
        if column in strict_data.columns
    ]
    if not searchable_columns:
        raise ValueError("Strict cohort has no auditable surgery text for obstetric exclusion")
    searchable_text = strict_data[searchable_columns].fillna("").astype(str).agg(" | ".join, axis=1)
    obstetric_mask = searchable_text.str.contains(OBSTETRIC_PATTERN, na=False)
    return strict_data.loc[~obstetric_mask].copy(), strict_data.loc[obstetric_mask].copy()


def first_icu_stay_per_admission(data: pd.DataFrame) -> pd.DataFrame:
    finalized = (
        data.sort_values(["hadm_id", "intime", "stay_id"], na_position="last")
        .drop_duplicates("hadm_id", keep="first")
        .copy()
    )
    if finalized["hadm_id"].duplicated().any():
        raise AssertionError("Final cohort is not unique by hadm_id")
    if finalized["stay_id"].duplicated().any():
        raise AssertionError("Final cohort is not unique by stay_id")
    finalized["analysis_unit"] = ANALYSIS_UNIT
    return finalized


def category_sets(series: pd.Series) -> pd.Series:
    def parse(value: object) -> set[str] | None:
        if pd.isna(value) or not str(value).strip():
            return None
        return {part.strip() for part in str(value).split("|") if part.strip()}

    return series.apply(parse)


def add_surgery_flags(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    if "surgery_categories" not in result.columns:
        result["surgery_categories"] = pd.NA
    parsed = category_sets(result["surgery_categories"])
    known = parsed.notna()
    for category, flag in CATEGORY_TO_FLAG.items():
        values = pd.Series(pd.NA, index=result.index, dtype="boolean")
        values.loc[known] = parsed.loc[known].apply(lambda groups: category in groups).astype(bool)
        result[flag] = values
    non_cardiac = pd.Series(pd.NA, index=result.index, dtype="boolean")
    non_cardiac.loc[known] = ~result.loc[known, "cardiac_surgery"].astype(bool)
    result["non_cardiac_surgery"] = non_cardiac
    return result


def enrich_broad_categories_from_strict(
    broad_data: pd.DataFrame, strict_source: pd.DataFrame
) -> pd.DataFrame:
    """Carry confirmed strict categories to matching B stay_ids; leave others unknown."""
    lookup = strict_source[["stay_id", "surgery_categories"]].drop_duplicates("stay_id")
    broad = broad_data.drop(columns=["surgery_categories"], errors="ignore")
    return broad.merge(lookup, on="stay_id", how="left", validate="one_to_one")


def safe_binary_summary(data: pd.DataFrame, column: str) -> tuple[int | None, float | None]:
    if column not in data.columns:
        return None, None
    numeric = pd.to_numeric(data[column], errors="coerce")
    observed = numeric.dropna()
    if observed.empty:
        return None, None
    return int((observed == 1).sum()), round(float((observed == 1).mean() * 100), 2)


def overall_summary(
    cohort_name: str,
    input_rows: int,
    finalized: pd.DataFrame,
    obstetric_excluded: int,
) -> dict[str, object]:
    aki_n, aki_rate = safe_binary_summary(finalized, "aki")
    los = pd.to_numeric(finalized["los"], errors="coerce")
    row: dict[str, object] = {
        "cohort": cohort_name,
        "analysis_unit": ANALYSIS_UNIT,
        "input_all_stays_n": input_rows,
        "obstetric_excluded_n": obstetric_excluded,
        "final_n": len(finalized),
        "unique_subjects_n": finalized["subject_id"].nunique(),
        "unique_hadm_id_n": finalized["hadm_id"].nunique(),
        "unique_stay_id_n": finalized["stay_id"].nunique(),
        "aki_n": aki_n,
        "aki_incidence_percent": aki_rate,
        "icu_los_observed_n": int(los.notna().sum()),
        "icu_los_mean_days": round(float(los.mean()), 3),
        "icu_los_sd_days": round(float(los.std()), 3),
        "icu_los_median_days": round(float(los.median()), 3),
        "icu_los_q1_days": round(float(los.quantile(0.25)), 3),
        "icu_los_q3_days": round(float(los.quantile(0.75)), 3),
        "icu_los_min_days": round(float(los.min()), 3),
        "icu_los_max_days": round(float(los.max()), 3),
    }
    for death_column in DEATH_COLUMNS:
        death_n, death_rate = safe_binary_summary(finalized, death_column)
        row[f"{death_column}_n"] = death_n
        row[f"{death_column}_percent"] = death_rate
    if "aki_stage" in finalized.columns:
        stages = pd.to_numeric(finalized["aki_stage"], errors="coerce")
        for stage in sorted(stages.dropna().unique()):
            stage_label = str(int(stage)) if float(stage).is_integer() else str(stage)
            row[f"aki_stage_{stage_label}_n"] = int((stages == stage).sum())
            row[f"aki_stage_{stage_label}_percent"] = round(float((stages == stage).mean() * 100), 2)
    return row


def distribution_rows(cohort_name: str, data: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def append_distribution(dimension: str, category: str, mask: pd.Series) -> None:
        subset = data.loc[mask]
        aki_n, aki_rate = safe_binary_summary(subset, "aki")
        rows.append(
            {
                "cohort": cohort_name,
                "dimension": dimension,
                "category": category,
                "n": len(subset),
                "percent_of_cohort": round(len(subset) / len(data) * 100, 2) if len(data) else None,
                "aki_n": aki_n,
                "aki_incidence_percent": aki_rate,
            }
        )

    for flag in [*CATEGORY_TO_FLAG.values(), "non_cardiac_surgery"]:
        if flag in data.columns:
            append_distribution("surgery_subgroup", flag, data[flag].fillna(False).astype(bool))

    if "first_careunit" in data.columns:
        for value in data["first_careunit"].fillna("<missing>").value_counts().index:
            append_distribution(
                "first_careunit",
                str(value),
                data["first_careunit"].fillna("<missing>").eq(value),
            )

    if "aki_stage" in data.columns:
        stage_text = data["aki_stage"].astype("string").fillna("<missing>")
        for value in stage_text.value_counts().index:
            append_distribution("aki_stage", str(value), stage_text.eq(value))

    return rows


def missingness_rows(cohort_name: str, data: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "cohort": cohort_name,
            "column": column,
            "dtype": str(data[column].dtype),
            "n_missing": int(data[column].isna().sum()),
            "missing_percent": round(float(data[column].isna().mean() * 100), 3),
            "n_observed": int(data[column].notna().sum()),
        }
        for column in data.columns
    ]


def write_readme(
    strict: pd.DataFrame,
    broad: pd.DataFrame,
    strict_input_n: int,
    broad_input_n: int,
    obstetric_excluded_n: int,
) -> None:
    strict_aki_n, strict_aki_rate = safe_binary_summary(strict, "aki")
    broad_aki_n, broad_aki_rate = safe_binary_summary(broad, "aki")
    broad_category_known = int(broad["surgery_categories"].notna().sum())
    content = f"""# Cohort v2 finalization audit

## Cohort definitions

- **Strict primary cohort (A):** strict postoperative surgical ICU screening cohort.
- **Broad sensitivity cohort (B):** broad procedure screening cohort.
- **Analysis unit:** `{ANALYSIS_UNIT}`. Rows were sorted by `hadm_id`, ICU `intime`, and `stay_id`; the earliest ICU stay was retained.
- `primary_surgery` was neither used nor retained in either finalized output.
- No raw MIMIC-IV table was read. The script reads only the two all-stays screening CSV files.
- No machine learning was performed.

## Final counts

| Cohort | Input stays | Obstetric exclusions | Final admissions | AKI n | AKI incidence |
|---|---:|---:|---:|---:|---:|
| A strict primary | {strict_input_n:,} | {obstetric_excluded_n:,} | {len(strict):,} | {strict_aki_n:,} | {strict_aki_rate:.2f}% |
| B broad sensitivity | {broad_input_n:,} | 0 | {len(broad):,} | {broad_aki_n:,} | {broad_aki_rate:.2f}% |

## Surgery subgroup flags

Flags were created only from `surgery_categories`. A row may belong to more than one specific surgical subgroup. `non_cardiac_surgery` means a known category set without `cardiac`; it is not the logical negation of missing information.

B did not originally contain `surgery_categories`. Confirmed categories were transferred only for exact `stay_id` matches found in A. B therefore has known categories for {broad_category_known:,} of {len(broad):,} rows; other B subgroup flags are missing/unknown rather than inferred from `primary_surgery`.

## Output interpretation

- `audit_v2_overall_summary.csv`: cohort size, AKI, AKI stage, mortality, and ICU LOS summaries.
- `audit_v2_surgery_subgroups.csv`: surgery-subgroup, ICU-type, and AKI-stage distributions with subgroup AKI incidence.
- `audit_v2_missingness_summary.csv`: column-level missingness for both finalized cohorts.

## Important limitation

The inherited AKI variables remain provisional screening labels. Final baseline creatinine and KDIGO event timing must be recomputed after the definitive index-time specification is locked.
"""
    (OUTPUT_DIR / "audit_v2_readme.md").write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    strict_input = load_input(INPUT_A, "Version A")
    broad_input = load_input(INPUT_B, "Version B")

    strict_non_obstetric, strict_obstetric = remove_obstetric_cases(strict_input)
    strict = first_icu_stay_per_admission(strict_non_obstetric)
    strict = add_surgery_flags(strict)

    broad_with_categories = enrich_broad_categories_from_strict(broad_input, strict_non_obstetric)
    broad = first_icu_stay_per_admission(broad_with_categories)
    broad = add_surgery_flags(broad)

    strict.insert(0, "cohort_v2", "strict_primary")
    broad.insert(0, "cohort_v2", "broad_sensitivity")

    strict.to_csv(OUTPUT_DIR / "cohort_v2_strict_primary.csv", index=False)
    broad.to_csv(OUTPUT_DIR / "cohort_v2_broad_sensitivity.csv", index=False)

    overall = pd.DataFrame(
        [
            overall_summary("strict_primary", len(strict_input), strict, len(strict_obstetric)),
            overall_summary("broad_sensitivity", len(broad_input), broad, 0),
        ]
    )
    overall.to_csv(OUTPUT_DIR / "audit_v2_overall_summary.csv", index=False)

    distributions = pd.DataFrame(
        distribution_rows("strict_primary", strict)
        + distribution_rows("broad_sensitivity", broad)
    )
    distributions.to_csv(OUTPUT_DIR / "audit_v2_surgery_subgroups.csv", index=False)

    missingness = pd.DataFrame(
        missingness_rows("strict_primary", strict)
        + missingness_rows("broad_sensitivity", broad)
    )
    missingness.to_csv(OUTPUT_DIR / "audit_v2_missingness_summary.csv", index=False)

    write_readme(
        strict,
        broad,
        strict_input_n=len(strict_input),
        broad_input_n=len(broad_input),
        obstetric_excluded_n=len(strict_obstetric),
    )

    print(overall.to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

