"""Prepare role-aware modeling tables and pre-modeling audits from v4 datasets.

This script does not split data and does not fit or evaluate any model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "outputs" / "dynamic_v4"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v4_1"

LANDMARKS = [0, 6, 24]
OUTCOME = "outcome_aki_after_landmark_to_7d"
ONSET_OUTCOME = "outcome_aki_onset_hours_from_icu"
METADATA_KEEP = ["subject_id", "hadm_id", "stay_id", "landmark_hours"]
NON_PREDICTOR_COLUMNS = {
    "subject_id", "hadm_id", "stay_id", "intime", "index_surgery_date",
    "landmark_hours", OUTCOME, ONSET_OUTCOME, "surgery_categories",
}
CATEGORICAL_ALLOWLIST = {
    "first_careunit", "gender", "race", "admission_type", "insurance",
    "marital_status", "baseline_scr_source_at_landmark",
}


def bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def require_columns(data: pd.DataFrame, columns: Iterable[str], landmark: int) -> None:
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"{landmark} h dataset is missing columns: {missing}")


def load_dataset(landmark: int) -> pd.DataFrame:
    path = INPUT_DIR / f"dataset_v4_{landmark}h.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path, low_memory=False)
    require_columns(data, [*METADATA_KEEP, "intime", "index_surgery_date", OUTCOME, ONSET_OUTCOME], landmark)
    if data["stay_id"].duplicated().any():
        raise ValueError(f"{landmark} h dataset contains duplicate stay_id")
    if set(pd.to_numeric(data["landmark_hours"], errors="coerce").dropna().unique()) != {landmark}:
        raise ValueError(f"{landmark} h dataset has an incorrect landmark_hours value")
    if data[OUTCOME].isna().any():
        raise ValueError(f"{landmark} h outcome contains missing values")
    return data


def looks_like_free_text(series: pd.Series, column: str) -> bool:
    if column in CATEGORICAL_ALLOWLIST:
        return False
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    observed = series.dropna().astype(str)
    if observed.empty:
        return False
    return observed.str.len().median() > 30 or observed.nunique() > 50


def select_predictors(data: pd.DataFrame, landmark: int) -> tuple[list[str], list[str]]:
    predictors: list[str] = []
    excluded: list[str] = []
    for column in data.columns:
        if column in NON_PREDICTOR_COLUMNS:
            excluded.append(column)
            continue
        if column.startswith("post_") or column.startswith("outcome_"):
            excluded.append(column)
            continue
        if looks_like_free_text(data[column], column):
            excluded.append(column)
            continue
        if landmark == 0 and ("_0_6h_" in column or "_0_24h_" in column):
            excluded.append(column)
            continue
        if landmark == 6 and "_0_24h_" in column:
            excluded.append(column)
            continue
        if landmark == 24 and "_0_6h_" in column:
            excluded.append(column)
            continue
        predictors.append(column)

    forbidden_fragments = ["aki_final", "aki_onset", "death", "expire", "outtime", "dischtime"]
    leaked = [
        column for column in predictors
        if column.startswith("post_") or any(fragment in column.lower() for fragment in forbidden_fragments)
    ]
    if leaked:
        raise ValueError(f"Leakage-prone predictors remain at {landmark} h: {leaked}")
    return predictors, sorted(set(excluded))


def infer_variable_type(series: pd.Series, column: str) -> str:
    if column in CATEGORICAL_ALLOWLIST:
        return "categorical"
    observed = series.dropna()
    if pd.api.types.is_bool_dtype(series):
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        unique = set(pd.to_numeric(observed, errors="coerce").dropna().unique())
        if unique and unique.issubset({0, 1}):
            return "binary"
        return "continuous_numeric"
    return "categorical"


def source_window(column: str, landmark: int) -> str:
    if column.startswith("lab_pre24h_"):
        return "pre-index_-24h_to_0h"
    if f"_0_{landmark}h_" in column:
        return f"post-index_0_to_{landmark}h"
    if column.endswith("_at_landmark"):
        return f"available_by_{landmark}h"
    return "baseline_or_index"


def level_counts(series: pd.Series) -> str:
    text = series.astype("string").fillna("<missing>")
    counts = text.value_counts(dropna=False)
    return " | ".join(f"{level}:{int(count)}" for level, count in counts.items())


def dictionary_rows(
    data: pd.DataFrame,
    landmark: int,
    predictors: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in METADATA_KEEP:
        rows.append(
            {
                "landmark_hours": landmark, "variable": column, "role": "metadata_not_predictor",
                "variable_type": "identifier", "source_window": "metadata", "n_observed": int(data[column].notna().sum()),
                "n_missing": int(data[column].isna().sum()), "missing_percent": round(float(data[column].isna().mean() * 100), 3),
                "n_unique": int(data[column].nunique(dropna=True)), "levels_or_distribution": "",
                "median": np.nan, "q1": np.nan, "q3": np.nan,
            }
        )
    for column in predictors:
        variable_type = infer_variable_type(data[column], column)
        observed = data[column].dropna()
        row: dict[str, object] = {
            "landmark_hours": landmark, "variable": column, "role": "predictor",
            "variable_type": variable_type, "source_window": source_window(column, landmark),
            "n_observed": int(data[column].notna().sum()), "n_missing": int(data[column].isna().sum()),
            "missing_percent": round(float(data[column].isna().mean() * 100), 3),
            "n_unique": int(data[column].nunique(dropna=True)), "levels_or_distribution": "",
            "median": np.nan, "q1": np.nan, "q3": np.nan,
        }
        if variable_type in {"categorical", "binary"}:
            row["levels_or_distribution"] = level_counts(data[column])
        else:
            numeric = pd.to_numeric(observed, errors="coerce")
            row["median"] = round(float(numeric.median()), 4) if numeric.notna().any() else np.nan
            row["q1"] = round(float(numeric.quantile(0.25)), 4) if numeric.notna().any() else np.nan
            row["q3"] = round(float(numeric.quantile(0.75)), 4) if numeric.notna().any() else np.nan
        rows.append(row)
    outcome = bool_mask(data[OUTCOME])
    rows.append(
        {
            "landmark_hours": landmark, "variable": OUTCOME, "role": "outcome",
            "variable_type": "binary", "source_window": f"after_{landmark}h_to_7d",
            "n_observed": len(data), "n_missing": 0, "missing_percent": 0.0,
            "n_unique": int(outcome.nunique()), "levels_or_distribution": level_counts(outcome),
            "median": np.nan, "q1": np.nan, "q3": np.nan,
        }
    )
    return rows


def write_readme(summary: pd.DataFrame, exclusions: dict[int, list[str]]) -> None:
    indexed = summary.set_index("landmark_hours")
    content = f"""# Modeling-ready dynamic AKI datasets v4.1

## Scope

- Binary outcome: `{OUTCOME}`.
- No train/test split, imputation, encoding, scaling, feature selection, or model fitting was performed.
- `subject_id`, `hadm_id`, and `stay_id` remain in the files only as metadata for later grouped splitting and traceability. They are not predictors.
- `landmark_hours` is metadata and is constant within each file.
- ICU/date fields, onset-time outcome, `surgery_categories`, free text, and leakage-prone future summaries were removed.

## Dataset summary

| Landmark | N | Events | Incidence | Predictors | Continuous | Binary | Categorical | >40% missing |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 h | {int(indexed.loc[0,'sample_size']):,} | {int(indexed.loc[0,'event_n']):,} | {indexed.loc[0,'event_incidence_percent']:.2f}% | {int(indexed.loc[0,'candidate_predictor_n']):,} | {int(indexed.loc[0,'continuous_numeric_n']):,} | {int(indexed.loc[0,'binary_n']):,} | {int(indexed.loc[0,'categorical_n']):,} | {int(indexed.loc[0,'predictors_missing_gt40pct_n']):,} |
| 6 h | {int(indexed.loc[6,'sample_size']):,} | {int(indexed.loc[6,'event_n']):,} | {indexed.loc[6,'event_incidence_percent']:.2f}% | {int(indexed.loc[6,'candidate_predictor_n']):,} | {int(indexed.loc[6,'continuous_numeric_n']):,} | {int(indexed.loc[6,'binary_n']):,} | {int(indexed.loc[6,'categorical_n']):,} | {int(indexed.loc[6,'predictors_missing_gt40pct_n']):,} |
| 24 h | {int(indexed.loc[24,'sample_size']):,} | {int(indexed.loc[24,'event_n']):,} | {indexed.loc[24,'event_incidence_percent']:.2f}% | {int(indexed.loc[24,'candidate_predictor_n']):,} | {int(indexed.loc[24,'continuous_numeric_n']):,} | {int(indexed.loc[24,'binary_n']):,} | {int(indexed.loc[24,'categorical_n']):,} | {int(indexed.loc[24,'predictors_missing_gt40pct_n']):,} |

## Variable roles

The variable dictionary is authoritative for downstream work:

- `metadata_not_predictor`: retained only for traceability and future patient-grouped splitting;
- `predictor`: candidate input variable;
- `outcome`: the binary target.

Categorical levels and binary counts are stored in `levels_or_distribution`. Continuous variables report median, Q1, and Q3. Missingness >40% is flagged but not automatically removed.

## Timing validation

- 0 h predictors contain baseline/index and pre-index variables only.
- 6 h dynamic predictors contain only `_0_6h_` windows.
- 24 h dynamic predictors contain only `_0_24h_` windows.
- Legacy `post_*`, AKI timing/stage, mortality, LOS, and unrestricted future summaries are absent.

## Excluded columns

- 0 h: `{'`, `'.join(exclusions[0])}`
- 6 h: `{'`, `'.join(exclusions[6])}`
- 24 h: `{'`, `'.join(exclusions[24])}`

No data split or machine learning was performed.
"""
    (OUTPUT_DIR / "audit_v4_1_readme.md").write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    missingness_rows: list[dict[str, object]] = []
    all_dictionary_rows: list[dict[str, object]] = []
    exclusions: dict[int, list[str]] = {}

    for landmark in LANDMARKS:
        data = load_dataset(landmark)
        predictors, excluded = select_predictors(data, landmark)
        exclusions[landmark] = excluded
        variable_types = {column: infer_variable_type(data[column], column) for column in predictors}

        # IDs are retained strictly for later grouped splitting; dates and auxiliary outcomes are dropped.
        modeling = data[[*METADATA_KEEP, *predictors, OUTCOME]].copy()
        modeling.to_csv(OUTPUT_DIR / f"modeling_v4_1_{landmark}h.csv", index=False)

        event = bool_mask(modeling[OUTCOME])
        missing_rates = modeling[predictors].isna().mean().mul(100)
        high_missing = missing_rates.loc[missing_rates > 40].sort_values(ascending=False)
        summary_rows.append(
            {
                "landmark_hours": landmark,
                "sample_size": len(modeling),
                "event_n": int(event.sum()),
                "event_incidence_percent": round(float(event.mean() * 100), 2),
                "candidate_predictor_n": len(predictors),
                "continuous_numeric_n": sum(value == "continuous_numeric" for value in variable_types.values()),
                "binary_n": sum(value == "binary" for value in variable_types.values()),
                "categorical_n": sum(value == "categorical" for value in variable_types.values()),
                "predictors_missing_gt40pct_n": len(high_missing),
                "predictors_missing_gt40pct": " | ".join(high_missing.index),
                "excluded_nonpredictor_or_leakage_n": len(excluded),
                "excluded_nonpredictor_or_leakage": " | ".join(excluded),
            }
        )
        for column in predictors:
            missingness_rows.append(
                {
                    "landmark_hours": landmark,
                    "predictor": column,
                    "variable_type": variable_types[column],
                    "n_missing": int(modeling[column].isna().sum()),
                    "missing_percent": round(float(modeling[column].isna().mean() * 100), 3),
                    "n_observed": int(modeling[column].notna().sum()),
                    "missing_gt40pct": bool(modeling[column].isna().mean() > 0.40),
                }
            )
        all_dictionary_rows.extend(dictionary_rows(modeling, landmark, predictors))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "audit_v4_1_modeling_ready_summary.csv", index=False)
    pd.DataFrame(missingness_rows).to_csv(
        OUTPUT_DIR / "audit_v4_1_predictor_missingness.csv", index=False
    )
    pd.DataFrame(all_dictionary_rows).to_csv(
        OUTPUT_DIR / "audit_v4_1_variable_dictionary.csv", index=False
    )
    write_readme(summary, exclusions)

    print(summary.drop(columns=["predictors_missing_gt40pct", "excluded_nonpredictor_or_leakage"]).to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

