"""Independent consistency checks for both v28 secondary-analysis modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SEVERE = ROOT / "outputs" / "modeling_v28_severe_temporal_external"
RECOVERY = ROOT / "outputs" / "modeling_v28_recovery_observability"
EICU = ROOT / "outputs" / "modeling_v16_eicu_external_validation" / "cohort_v16_eicu_external_validation.csv.gz"
V27 = ROOT / "outputs" / "modeling_v27_severity_recovery"
LANDMARKS = [0, 6, 24]


def bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).astype(bool)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def close(actual: float, expected: float, tolerance: float = 1e-10) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=tolerance, equal_nan=True):
        raise AssertionError(f"{actual} != {expected}")


def validate_severe() -> list[str]:
    checks: list[str] = []
    cohort = pd.read_csv(EICU, low_memory=False)
    states = pd.read_csv(SEVERE / "eicu_creatinine_measurement_states_v28.csv.gz")
    onsets = pd.read_csv(SEVERE / "eicu_severe_onset_rrt_v28.csv")
    predictions = pd.read_csv(SEVERE / "model_v28_eicu_frozen_severe_predictions.csv")
    performance = pd.read_csv(SEVERE / "model_v28_eicu_frozen_severe_performance.csv")

    if ((states["aki_stage_at_measurement"] > 0) & ~bool_mask(states["aki_active"])).any():
        raise AssertionError("Non-active SCr measurement assigned an AKI stage")
    stage3 = states["aki_stage_at_measurement"].eq(3)
    qualifying3 = states["scr_ratio_to_baseline"].ge(3) | states["scr_mg_dl"].ge(4)
    if (stage3 & ~qualifying3).any():
        raise AssertionError("Stage 3 measurement lacks ratio >=3 or active SCr >=4 mg/dL")
    severe_by_stay = states.groupby("stay_id")["aki_stage_at_measurement"].max().ge(2)
    onset_severe = onsets.set_index("stay_id")["severe_scr_onset_hours"].notna()
    if not severe_by_stay.reindex(onset_severe.index, fill_value=False).equals(onset_severe):
        raise AssertionError("Severe onset table disagrees with measurement-level maximum stage")
    checks.append("Measurement-level active KDIGO staging and first severe onset agree")

    evaluable = cohort.loc[bool_mask(cohort["incident_aki_evaluable"]), ["stay_id"]].merge(
        onsets, on="stay_id", validate="one_to_one"
    )
    for landmark in LANDMARKS:
        data = pd.read_csv(SEVERE / f"dataset_v28_eicu_severe_{landmark}h.csv.gz", low_memory=False)
        expected_primary = evaluable.loc[
            evaluable["severe_scr_onset_hours"].isna()
            | evaluable["severe_scr_onset_hours"].gt(landmark)
        ]
        if len(data) != len(expected_primary) or set(data["stay_id"]) != set(expected_primary["stay_id"]):
            raise AssertionError(f"Incorrect primary severe risk set at {landmark} h")
        expected_events = int(expected_primary["severe_scr_onset_hours"].gt(landmark).sum())
        if int(data["outcome_severe_scr_after_landmark_to_7d"].sum()) != expected_events:
            raise AssertionError(f"Incorrect primary severe outcome at {landmark} h")

        expected_combined = expected_primary.loc[
            expected_primary["severe_scr_or_rrt_onset_hours"].isna()
            | expected_primary["severe_scr_or_rrt_onset_hours"].gt(landmark)
        ]
        pred_combined = predictions.loc[
            predictions["landmark_hours"].eq(landmark)
            & predictions["target"].eq("SCr stage 2/3 or RRT sensitivity")
        ]
        if len(pred_combined) != len(expected_combined) or set(pred_combined["stay_id"]) != set(expected_combined["stay_id"]):
            raise AssertionError(f"Incorrect combined SCr/RRT risk set at {landmark} h")
        if int(pred_combined["outcome_severe_aki"].sum()) != int(
            expected_combined["severe_scr_or_rrt_onset_hours"].gt(landmark).sum()
        ):
            raise AssertionError(f"Incorrect combined outcome at {landmark} h")
    checks.append("All eICU landmark-specific SCr and SCr/RRT risk sets and outcomes reproduce")

    for row in performance.itertuples():
        subset = predictions.loc[
            predictions["landmark_hours"].eq(row.landmark_hours)
            & predictions["target"].eq(row.target)
        ]
        y = subset["outcome_severe_aki"].astype(int)
        p = subset["predicted_risk_frozen"]
        close(roc_auc_score(y, p), row.auroc)
        close(average_precision_score(y, p), row.auprc)
        close(brier_score_loss(y, p), row.brier_score)
    checks.append("Frozen external AUROC, AUPRC, and Brier scores reproduce from predictions")

    membership = pd.read_csv(SEVERE / "audit_v28_hospital_partition_membership.csv")
    if membership["hospitalid"].duplicated().any() or set(membership["partition"]) != {"recalibration", "held-out validation"}:
        raise AssertionError("Hospital partition is not mutually exclusive and exhaustive")
    heldout = pd.read_csv(SEVERE / "model_v28_heldout_hospital_predictions.csv")
    if not set(heldout["hospitalid"]).issubset(
        set(membership.loc[membership["partition"].eq("held-out validation"), "hospitalid"])
    ):
        raise AssertionError("Held-out predictions contain recalibration hospitals")
    checks.append("Hospital-level recalibration and validation partitions are disjoint")

    dca = pd.read_csv(SEVERE / "analysis_v28_severe_decision_curve.csv")
    expected_methods = {
        "treat all", "treat none", "prior stage 1 rule", "CKD rule", "model",
        "frozen model", "logistic recalibration",
    }
    if not set(dca["method"]).issubset(expected_methods) or dca["net_benefit"].isna().any():
        raise AssertionError("DCA method or net-benefit output is invalid")
    checks.append("Decision-curve outputs are finite and use prespecified comparators")
    return checks


def validate_recovery() -> list[str]:
    checks: list[str] = []
    onset = pd.read_csv(V27 / "dataset_v27_aki_onset_trajectory.csv", low_memory=False)
    oof = pd.read_csv(RECOVERY / "model_v28_observability_oof_predictions.csv")
    if len(oof) != len(onset) or oof["stay_id"].duplicated().any():
        raise AssertionError("Observability OOF prediction coverage is incorrect")
    for task, evaluable in [
        ("persistent_aki", "persistence_evaluable"),
        ("nonrecovery", "end_recovery_evaluable"),
    ]:
        probability = oof[f"{task}_observability_probability_oof"]
        if probability.isna().any() or (~probability.between(0, 1)).any():
            raise AssertionError(f"Invalid observability probability for {task}")
        if not bool_mask(onset[evaluable]).reset_index(drop=True).equals(
            bool_mask(oof[f"{task}_evaluable"]).reset_index(drop=True)
        ):
            raise AssertionError(f"Evaluability flag mismatch for {task}")
    checks.append("Cross-fitted observability predictions cover every AKI-onset stay")

    candidate = pd.read_csv(V27 / "audit_v27_predictor_missingness.csv")
    candidate = candidate.loc[candidate["dataset"].eq("aki_onset"), "predictor"].str.lower()
    forbidden = ("outcome", "recovered", "persistent", "evaluable", "trajectory_group")
    if candidate.map(lambda x: any(token in x for token in forbidden)).any():
        raise AssertionError("Observability model includes a post-onset outcome/evaluability predictor")
    checks.append("Observability predictors are restricted to AKI-onset information")

    competing = pd.read_csv(RECOVERY / "audit_v28_recovery_competing_events.csv")
    for task in ["persistent_aki", "nonrecovery"]:
        subset = competing.loc[competing["task"].eq(task)]
        if int(subset["n"].sum()) != len(onset):
            raise AssertionError(f"Competing-state counts do not exhaust cohort for {task}")
        close(subset["percent"].sum(), 100.0, 1e-8)
    checks.append("Observed, death, live-discharge, and monitoring-gap states exhaust the cohort")

    ipw = pd.read_csv(RECOVERY / "model_v28_recovery_ipw_performance.csv")
    original_predictions = pd.read_csv(V27 / "model_v27_test_predictions.csv")
    for task, model in [("persistent_aki", "xgboost"), ("nonrecovery", "logistic_regression")]:
        subset = original_predictions.loc[original_predictions["task"].eq(task)]
        observed_auc = roc_auc_score(subset["outcome"], subset[f"probability_{model}"])
        reported = ipw.loc[
            ipw["task"].eq(task) & ipw["analysis"].eq("complete case"), "auroc"
        ].iloc[0]
        close(observed_auc, reported)
    checks.append("Complete-case recovery metrics reproduce the locked v27 held-out predictions")

    composite = pd.read_csv(RECOVERY / "model_v28_adverse_composite_test_predictions.csv")
    performance = pd.read_csv(RECOVERY / "model_v28_adverse_composite_performance.csv")
    for row in performance.itertuples():
        subset = composite.loc[composite["task"].eq(row.task)]
        y = pd.to_numeric(subset[row.outcome], errors="raise").astype(int)
        close(roc_auc_score(y, subset["predicted_risk"]), row.auroc)
        close(average_precision_score(y, subset["predicted_risk"]), row.auprc)
    checks.append("Adverse-composite held-out AUROC and AUPRC reproduce")
    return checks


def main() -> None:
    checks = validate_severe() + validate_recovery()
    lines = ["# Independent v28 validation", "", f"Status: PASS ({len(checks)} checks)", ""]
    lines += [f"- {check}" for check in checks]
    report = "\n".join(lines) + "\n"
    (SEVERE / "audit_v28_independent_validation_report.md").write_text(report, encoding="utf-8")
    (RECOVERY / "audit_v28_independent_validation_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
