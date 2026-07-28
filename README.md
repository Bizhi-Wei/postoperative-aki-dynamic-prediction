# Dynamic prediction of incident postoperative AKI in surgical intensive care

Reproducible analytic code for a retrospective MIMIC-IV and eICU cohort study of dynamic prediction of serum-creatinine-defined postoperative acute kidney injury (AKI) at ICU admission, 6 hours, and 24 hours.

The final locked manuscript reports a strict postoperative surgical ICU cohort, a timestamped KDIGO serum-creatinine incident-AKI outcome, landmark-specific datasets, internal and temporal validation, eICU feature-harmonized external validation, hospital-heldout recalibration, competing-risk sensitivity analyses, and assumption-driven selection-bias analyses.

The post-lock v26 secondary phenotype analysis additionally characterizes maximum AKI severity, rapid reversal, persistent AKI beyond 48 hours, recurrent AKI, and observed seven-day/discharge recovery. It preserves explicit persistence and recovery evaluability flags so that missing repeat creatinine measurements are not interpreted as recovery.

The post-lock v27 secondary modeling analysis predicts new active-episode SCr stage 2/3 AKI at 0, 6, and 24 hours while retaining stage 1 patients in the later risk sets. Separate onset-anchored models predict AKI persistence beyond 48 hours and nonrecovery at the observed seven-day/discharge end point; these recovery estimates are explicitly conditional on follow-up SCr observability.

The v31 audit evaluates model-family selection entirely within the original 80% training partition using five outer and four inner stratified subject-grouped folds. The corresponding manuscript supplement also consolidates the existing eICU outcome-observability comparison, cross-fitted inverse-probability-weighted and pattern-mixture analyses, and strict pre-ICU baseline-creatinine sensitivity analysis.

## Scope and key design choices

- Analysis unit: first qualifying ICU stay per hospital admission.
- Primary outcome: incident KDIGO serum-creatinine AKI within 7 days after ICU admission; urine output is not part of the primary outcome.
- Landmarks: 0, 6, and 24 hours; patients with AKI already present at a landmark are excluded from that landmark risk set.
- Development cohort: strict therapeutic postoperative surgical ICU cohort in MIMIC-IV v3.1.
- External evaluation: feature-harmonized portable models in eICU; this is not a direct validation of every full MIMIC-IV predictor.
- Deployment boundary: model artifacts are research-use candidates only. They do not provide a universal action threshold or an automated treatment recommendation.

## Repository contents

- `scripts/`: cohort construction, outcome derivation, dynamic feature engineering, model development, validation, sensitivity, manuscript, and research-use deployment scripts.
- `scripts/severe_persistent_aki_trajectories_v26.py`: serum-creatinine severity, persistence, recurrence, end-of-window recovery, RRT-overlay, observability, and descriptive outcome analysis.
- `scripts/validate_severe_persistent_aki_trajectories_v26.py`: independent row-level consistency and audit-table reconciliation for v26.
- `scripts/secondary_severity_recovery_models_v27.py`: severe-AKI dynamic risk sets and onset-anchored persistence/nonrecovery model development with grouped internal validation.
- `scripts/validate_secondary_severity_recovery_models_v27.py`: independent risk-set, timing, leakage, split, and metric validation for v27.
- `scripts/training_only_nested_grouped_cv_v31.py`: leakage-controlled training-only nested grouped model-family selection audit.
- `scripts/build_secondary_manuscript_v31.py`: manuscript and supplementary package builder with Tables S1–S12.
- `scripts/validate_secondary_manuscript_v31.py`: independent scientific, numbering, source-value, and package-consistency checks for v31.
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

The frozen reported model specifications are XGBoost with 36 predictors at 0 hours, XGBoost with 72 predictors at 6 hours, and logistic regression with 72 predictors at 24 hours. The retrospective training-only AUROC audit preferred XGBoost at all three landmarks, but the 24-hour training OOF AUROC margin over logistic regression was only 0.001. The manuscript reports this instability and retains the previously frozen 24-hour logistic external analysis rather than switching models after the audit. Before any implementation outside the development setting, prospective silent validation, local probability recalibration, data-quality mapping, governance review, and a clinician-approved response pathway are required.

Selection-bias analyses quantify robustness under stated missing-data assumptions; they do not establish that outcome-observability bias has been removed. SHAP values and secondary adjusted associations are predictive/prognostic quantities, not causal effects or proof of modifiability.

## Citation

Please cite the associated manuscript and software release v1.0.4. The software archive DOI will be added after the Zenodo archive is published.
