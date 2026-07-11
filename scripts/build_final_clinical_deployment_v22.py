"""Build a research-use deployment candidate for dynamic postoperative AKI prediction.

This script freezes the v14 selected parsimonious model specifications and
refits those specifications on all eligible MIMIC-IV development data.  It is
not a clinical decision-support release: no universal action threshold is
created, and local prospective validation and recalibration remain required.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from develop_models_v5 import (  # noqa: E402
    METADATA, OUTCOME, RANDOM_STATE, identify_types, load_data, make_preprocessor,
)

PROJECT_ROOT = SCRIPT_DIR.parent
V14_DIR = PROJECT_ROOT / "outputs" / "modeling_v14_final_sensitivities"
OUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v22_final_clinical_deployment"
MODEL_DIR = OUT_DIR / "models"
LANDMARKS = [0, 6, 24]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binary_to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    values = series.astype("string").str.strip().str.lower()
    return values.map({"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "1": 1.0, "0": 0.0})


def build_pipeline(model_family: str, continuous: list[str], binary: list[str], categorical: list[str]) -> Pipeline:
    if model_family == "XGBoost":
        estimator = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=4,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE,
        )
        scale = False
    elif model_family == "Logistic Regression":
        estimator = LogisticRegression(max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE)
        scale = True
    else:
        raise ValueError(f"Unsupported selected model family: {model_family}")
    return Pipeline([("preprocess", make_preprocessor(continuous, binary, categorical, scale=scale)), ("model", estimator)])


def reference_rows(data: pd.DataFrame, predictors: list[str], continuous: list[str], binary: list[str], categorical: list[str], landmark: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for col in predictors:
        s = data[col]
        row: dict[str, object] = {
            "landmark_hours": landmark, "predictor": col,
            "variable_type": "continuous" if col in continuous else "binary" if col in binary else "categorical",
            "development_n": len(s), "missing_n": int(s.isna().sum()), "missing_percent": float(s.isna().mean() * 100),
            "median": np.nan, "p05": np.nan, "p95": np.nan, "levels_or_distribution": "",
        }
        if col in continuous:
            v = pd.to_numeric(s, errors="coerce")
            row.update({"median": float(v.median()), "p05": float(v.quantile(.05)), "p95": float(v.quantile(.95))})
        elif col in binary:
            v = binary_to_numeric(s)
            row["levels_or_distribution"] = f"proportion_1={v.mean():.6f}"
        else:
            levels = s.astype("string").fillna("<MISSING>").value_counts(dropna=False)
            row["levels_or_distribution"] = "; ".join(f"{k}={int(v)}" for k, v in levels.items())
        rows.append(row)
    return rows


def write_model_card(manifest: dict[str, object]) -> None:
    lines = [
        "# Dynamic postoperative AKI prediction: research-use deployment candidate (v22)",
        "",
        "## Intended use", "",
        "This package estimates the probability of new serum-creatinine KDIGO AKI from each landmark to 7 days in the strict postoperative ICU cohort. It is a research-use implementation artifact, not an approved clinical decision-support system.",
        "",
        "## Fixed model specifications", "",
        "| Landmark | Model | Predictors | Outcome risk window |",
        "|---:|---|---:|---|",
    ]
    for lm in LANDMARKS:
        spec = manifest["models"][str(lm)]
        lines.append(f"| {lm} h | {spec['model_family']} | {len(spec['predictors'])} | after {lm} h through 7 days after ICU admission |")
    lines += [
        "",
        "## Eligibility before scoring", "",
        "- Strict postoperative ICU population; first ICU stay within the admission.",
        "- Patient remains in ICU at the requested landmark.",
        "- Baseline serum creatinine is available and the patient has not already met AKI criteria at or before that landmark.",
        "- At 6 h and 24 h, do not score patients who have already developed AKI; these are incident-AKI risk-set models.",
        "",
        "## Output and actions", "",
        "Return the continuous predicted probability only. Do not automatically order tests, trigger treatment, or deny care from this score. A site may establish a review threshold only after prospective silent validation, local calibration assessment, clinician co-design, and governance approval.",
        "",
        "## Validation evidence and limitations", "",
        "The selected parsimonious specifications had held-out MIMIC-IV AUROC 0.726, 0.736, and 0.756 at 0 h, 6 h, and 24 h, respectively. eICU validation showed transportable but lower discrimination and material calibration heterogeneity; therefore, a local recalibration update is mandatory before any actionable use.",
        "Creatinine-record observability is selective. The v21 IPW and pattern-mixture analyses provide robustness bounds, not proof that selection bias has been removed. The outcome does not yet use urine output in the primary deployment model.",
        "",
        "## Monitoring and change control", "",
        "- Run silently first; monitor eligibility, missingness, score distribution, calibration-in-the-large, calibration slope, AUROC/AUPRC, and care-process effects by landmark and key subgroups.",
        "- Compare each input feature against `model_v22_reference_data_profile.csv`; investigate material missingness or distribution shifts before use.",
        "- Reassess at least quarterly and after EHR, laboratory, or surgical-workflow changes. Version every model, feature mapping, calibration update, and threshold policy.",
        "- Do not reuse the supplied MIMIC-trained probability scale at another hospital without an approved local recalibration study.",
        "",
        "## Reproducibility", "",
        "Model artifacts are fitted on all MIMIC-IV development records after model family and predictor set selection. This refit does not replace the previously reported internal, temporal, and external validation results.",
    ]
    (OUT_DIR / "model_card_v22.md").write_text("\n".join(lines), encoding="utf-8")


def write_scoring_readme() -> None:
    content = """# Scoring interface

Example:

```powershell
python scripts\\score_deployment_model_v22.py --landmark 6 --input path\\to\\eligible_6h_patients.csv --output scored_6h_patients.csv
```

The input must contain every predictor listed in `model_v22_predictor_dictionary.csv` for that landmark and all four non-model eligibility fields in the supplied template. The latter are not passed into the model, but are required to prevent prediction outside the modeled incident-AKI risk set. Rows failing eligibility are retained for audit and receive no risk probability. Missing predictor values are handled only by the fitted development preprocessing pipeline. The scoring script cannot verify the clinical correctness, units, timestamps, or upstream EHR feature mapping.
"""
    (OUT_DIR / "SCORING_README.md").write_text(content, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(V14_DIR / "model_v14_selected_parsimonious_model_finalization.csv")
    manifest: dict[str, object] = {
        "package_version": "v22", "created_date": str(date.today()), "intended_use": "research-use deployment candidate only",
        "outcome": "incident serum-creatinine KDIGO AKI after landmark through 7 days after ICU admission",
        "analysis_unit": "first ICU stay per hospital admission", "models": {},
        "eligibility_fields_not_model_predictors": ["is_first_icu_stay_per_admission", "patient_in_icu_at_landmark", "has_baseline_scr", "preexisting_aki_at_or_before_landmark"],
    }
    dictionary_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []

    for lm in LANDMARKS:
        selection = selected.loc[selected["landmark_hours"].eq(lm)].iloc[0]
        predictors = [v.strip() for v in str(selection["predictors"]).split(";") if v.strip()]
        data = load_data(lm)
        absent = sorted(set(predictors) - set(data.columns))
        if absent:
            raise ValueError(f"{lm} h selected predictors missing from source dataset: {absent}")
        model_data = data[[*METADATA, OUTCOME, *predictors]].copy()
        continuous, binary, categorical = identify_types(model_data)
        pipeline = build_pipeline(str(selection["final_model_family"]), continuous, binary, categorical)
        pipeline.fit(data[predictors], data[OUTCOME].to_numpy(dtype=int))
        artifact = MODEL_DIR / f"aki_dynamic_{lm}h_{str(selection['final_model_family']).lower().replace(' ', '_')}.joblib"
        joblib.dump(pipeline, artifact, compress=3)
        # Artifact round-trip check catches serialization or schema corruption.
        loaded = joblib.load(artifact)
        before = pipeline.predict_proba(data.loc[:9, predictors])[:, 1]
        after = loaded.predict_proba(data.loc[:9, predictors])[:, 1]
        if not np.allclose(before, after, rtol=0, atol=1e-12):
            raise AssertionError(f"{lm} h artifact prediction round-trip failed")
        manifest["models"][str(lm)] = {
            "landmark_hours": lm, "model_family": str(selection["final_model_family"]), "predictors": predictors,
            "continuous_predictors": continuous, "binary_predictors": binary, "categorical_predictors": categorical,
            "artifact": str(artifact.relative_to(OUT_DIR)).replace("\\", "/"), "artifact_sha256": sha256(artifact),
            "development_n_refit": int(len(data)), "development_event_rate_refit": float(data[OUTCOME].mean()),
            "model_selection_validation": {
                "heldout_auroc": float(selection["auroc"]), "heldout_auroc_95ci": [float(selection["auroc_ci_lower"]), float(selection["auroc_ci_upper"])],
                "heldout_auprc": float(selection["auprc"]), "heldout_brier": float(selection["brier_score"]),
            },
        }
        for col in predictors:
            dictionary_rows.append({"landmark_hours": lm, "predictor": col, "variable_type": "continuous" if col in continuous else "binary" if col in binary else "categorical", "model_predictor": True, "availability_window": "at/before ICU admission" if lm == 0 else f"at/before {lm} h after ICU admission"})
        profile_rows.extend(reference_rows(data, predictors, continuous, binary, categorical, lm))
        template = pd.DataFrame(columns=[*manifest["eligibility_fields_not_model_predictors"], *predictors])
        template.to_csv(OUT_DIR / f"deployment_input_template_{lm}h.csv", index=False)
        training_rows.append({"landmark_hours": lm, "model_family": str(selection["final_model_family"]), "predictor_n": len(predictors), "refit_n": len(data), "refit_event_n": int(data[OUTCOME].sum()), "refit_event_rate": float(data[OUTCOME].mean()), "artifact": artifact.name, "artifact_sha256": sha256(artifact), "serialization_roundtrip_passed": True})

    (OUT_DIR / "model_v22_deployment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(dictionary_rows).to_csv(OUT_DIR / "model_v22_predictor_dictionary.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(OUT_DIR / "model_v22_reference_data_profile.csv", index=False)
    pd.DataFrame(training_rows).to_csv(OUT_DIR / "audit_v22_model_artifact_summary.csv", index=False)
    write_model_card(manifest)
    write_scoring_readme()
    print(f"Wrote final clinical deployment candidate to: {OUT_DIR}")


if __name__ == "__main__":
    main()
