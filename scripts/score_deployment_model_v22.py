"""Score an eligible CSV with a v22 research-use deployment model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def as_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def eligibility_status(row: pd.Series) -> str:
    checks = {
        "is_first_icu_stay_per_admission": True,
        "patient_in_icu_at_landmark": True,
        "has_baseline_scr": True,
        "preexisting_aki_at_or_before_landmark": False,
    }
    supplied = 0
    for column, expected in checks.items():
        if column in row.index:
            got = as_bool(row[column])
            if got is not None:
                supplied += 1
                if got != expected:
                    return "ineligible_do_not_score"
    return "eligible_checks_complete" if supplied == len(checks) else "eligibility_not_fully_assessed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score v22 research-use postoperative AKI model")
    parser.add_argument("--landmark", required=True, type=int, choices=[0, 6, 24])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "modeling_v22_final_clinical_deployment")
    args = parser.parse_args()
    manifest = json.loads((args.package_dir / "model_v22_deployment_manifest.json").read_text(encoding="utf-8"))
    spec = manifest["models"][str(args.landmark)]
    data = pd.read_csv(args.input, low_memory=False)
    predictors = spec["predictors"]
    eligibility_fields = manifest["eligibility_fields_not_model_predictors"]
    missing = sorted((set(predictors) | set(eligibility_fields)) - set(data.columns))
    if missing:
        raise ValueError(f"Input misses required predictor or eligibility columns: {missing}")
    model = joblib.load(args.package_dir / spec["artifact"])
    probabilities = model.predict_proba(data[predictors])[:, 1]
    result = data.copy()
    result["prediction_landmark_hours"] = args.landmark
    result["eligibility_status"] = data.apply(eligibility_status, axis=1)
    # Never release a probability for a patient explicitly outside the modeled
    # incident-AKI risk set. Retain the row and status for implementation audit.
    result["predicted_incident_aki_risk_to_7d"] = np.where(
        result["eligibility_status"].eq("eligible_checks_complete"), probabilities, np.nan
    )
    result["deployment_notice"] = "research-use probability only; no automatic action threshold"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result):,} rows; wrote {args.output}")
    ineligible = int(result["eligibility_status"].ne("eligible_checks_complete").sum())
    if ineligible:
        print(f"Warning: {ineligible:,} ineligible rows were retained for audit and returned no risk probability.")


if __name__ == "__main__":
    main()
