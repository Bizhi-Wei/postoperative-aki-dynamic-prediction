# Dynamic prediction of incident postoperative AKI in surgical intensive care

Reproducible analytic code for a retrospective MIMIC-IV and eICU cohort study of dynamic prediction of serum-creatinine-defined postoperative acute kidney injury (AKI) at ICU admission, 6 hours, and 24 hours.

The final locked manuscript reports a strict postoperative surgical ICU cohort, a timestamped KDIGO serum-creatinine incident-AKI outcome, landmark-specific datasets, internal and temporal validation, eICU feature-harmonized external validation, hospital-heldout recalibration, competing-risk sensitivity analyses, and assumption-driven selection-bias analyses.

## Scope and key design choices

- Analysis unit: first qualifying ICU stay per hospital admission.
- Primary outcome: incident KDIGO serum-creatinine AKI within 7 days after ICU admission; urine output is not part of the primary outcome.
- Landmarks: 0, 6, and 24 hours; patients with AKI already present at a landmark are excluded from that landmark risk set.
- Development cohort: strict therapeutic postoperative surgical ICU cohort in MIMIC-IV v3.1.
- External evaluation: feature-harmonized portable models in eICU; this is not a direct validation of every full MIMIC-IV predictor.
- Deployment boundary: model artifacts are research-use candidates only. They do not provide a universal action threshold or an automated treatment recommendation.

## Repository contents

- `scripts/`: cohort construction, outcome derivation, dynamic feature engineering, model development, validation, sensitivity, manuscript, and research-use deployment scripts.
- `docs/`: cohort and reproducibility specifications.
- `requirements.txt`: Python package requirements.
- `.zenodo.json`: metadata for a release archive.

No MIMIC-IV/eICU source data, patient-level analytic datasets, model predictions, binary model artifacts, rendered manuscript files, or QA material are included in this repository.

## Data access and setup

MIMIC-IV v3.1 and eICU-CRD are available only to credentialed users through PhysioNet and under their respective data-use agreements. Users must download data independently and must not commit or redistribute it through this repository.

Set local data roots before running scripts that access the protected source tables:

```powershell
$env:MIMIC_IV_ROOT = 'D:\path\to\mimic-iv-3.1'
$env:EICU_ROOT = 'D:\path\to\eicu-collaborative-research-database-2.0'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run scripts in numbered analytic order. Later stages read the versioned aggregate/intermediate outputs created by preceding stages; they do not download data automatically.

## Reproducibility and limitations

The final model specifications are XGBoost with 36 predictors at 0 hours, XGBoost with 72 predictors at 6 hours, and logistic regression with 72 predictors at 24 hours. Before any implementation outside the development setting, prospective silent validation, local probability recalibration, data-quality mapping, governance review, and a clinician-approved response pathway are required.

Selection-bias analyses quantify robustness under stated missing-data assumptions; they do not establish that outcome-observability bias has been removed. SHAP values and secondary adjusted associations are predictive/prognostic quantities, not causal effects or proof of modifiability.

## Citation

Please cite the associated manuscript and the archived software release. The software archive DOI will be added after the first Zenodo release is published.
