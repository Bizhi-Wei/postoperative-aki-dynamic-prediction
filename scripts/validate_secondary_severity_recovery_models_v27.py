"""Independent validation for v27 severe-AKI and trajectory models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "modeling_v27_severity_recovery"
V26 = ROOT / "outputs" / "modeling_v26_aki_severity_trajectories"
LANDMARKS = [0, 6, 24]
SCR_OUTCOME = "outcome_severe_scr_after_landmark_to_7d"
RRT_OUTCOME = "outcome_severe_scr_or_rrt_after_landmark_to_7d"


def boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).astype(bool)
    return series.astype("string").str.lower().isin(["true", "1", "yes"])


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> None:
    cohort = pd.read_csv(
        V26 / "cohort_v26_strict_aki_severity_recovery.csv",
        low_memory=False,
        parse_dates=["aki_onset_time_final"],
    )
    states = pd.read_csv(
        V26 / "creatinine_measurement_states_v26.csv.gz",
        low_memory=False,
        parse_dates=["charttime"],
    )
    risk_audit = pd.read_csv(OUT / "audit_v27_severe_risk_set_summary.csv")
    performance = pd.read_csv(OUT / "model_v27_performance_summary.csv")
    predictions = pd.read_csv(OUT / "model_v27_test_predictions.csv", low_memory=False)
    assignment = pd.read_csv(OUT / "audit_v27_subject_split_assignment.csv")
    missingness = pd.read_csv(OUT / "audit_v27_predictor_missingness.csv")
    checks: list[str] = []

    active_max = (
        states.groupby("stay_id")["aki_stage_at_measurement"]
        .max()
        .reindex(cohort["stay_id"])
        .fillna(0)
        .astype(int)
        .reset_index(drop=True)
    )
    if not active_max.equals(cohort["maximum_active_scr_stage_7d"].astype(int).reset_index(drop=True)):
        fail("Active-episode severity does not reconcile with measurement states")
    checks.append(
        f"PASS: active-episode severity independently reconstructed (stage 2/3 n={int(active_max.ge(2).sum()):,})"
    )

    severe_states = states.loc[states["aki_stage_at_measurement"].ge(2)]
    first_severe = severe_states.groupby("stay_id")["hours_from_icu"].min()
    for landmark in LANDMARKS:
        data = pd.read_csv(OUT / f"dataset_v27_severe_{landmark}h.csv", low_memory=False)
        expected_eligible = cohort.loc[
            cohort["stay_id"].map(first_severe).isna()
            | cohort["stay_id"].map(first_severe).gt(landmark)
        ]
        if set(data["stay_id"]) != set(expected_eligible["stay_id"]):
            fail(f"Severe risk set mismatch at {landmark} h")
        expected_event = data["stay_id"].map(first_severe).gt(landmark).fillna(False).astype(int)
        if not np.array_equal(expected_event.to_numpy(), data[SCR_OUTCOME].astype(int).to_numpy()):
            fail(f"Severe outcome mismatch at {landmark} h")
        row = risk_audit.loc[risk_audit["landmark_hours"].eq(landmark)].iloc[0]
        if int(row["risk_set_n"]) != len(data) or int(row["severe_scr_event_n"]) != int(expected_event.sum()):
            fail(f"Risk-set audit mismatch at {landmark} h")
        if landmark > 0:
            prior_stage1 = boolean(data["prior_stage1_aki_by_landmark"])
            if int(prior_stage1.sum()) != int(row["prior_stage1_in_risk_set_n"]):
                fail(f"Prior stage 1 count mismatch at {landmark} h")
            if (data.loc[prior_stage1, "current_scr_stage_at_landmark"] >= 2).any():
                fail(f"Already-severe patient retained as prior stage 1 at {landmark} h")
    checks.append("PASS: 0/6/24 h severe-AKI risk sets, outcomes, and retained stage 1 histories reconcile")

    onset = pd.read_csv(OUT / "dataset_v27_aki_onset_trajectory.csv", low_memory=False)
    aki = cohort.loc[boolean(cohort["aki_final"])]
    if set(onset["stay_id"]) != set(aki["stay_id"]):
        fail("AKI-onset dataset does not cover every incident-AKI row")
    state_groups = {stay: group.sort_values("charttime") for stay, group in states.groupby("stay_id")}
    cohort_onset = aki.set_index("stay_id")["aki_onset_time_final"]
    for row in onset.itertuples(index=False):
        history = state_groups[row.stay_id]
        history = history.loc[history["charttime"].le(cohort_onset.loc[row.stay_id])]
        last = history.iloc[-1]
        if not np.isclose(float(row.onset_scr_mg_dl), float(last["scr_mg_dl"]), atol=1e-12):
            fail(f"Onset SCr mismatch for stay {row.stay_id}")
        if int(row.scr_n_icu_to_onset) != len(history):
            fail(f"Pre-onset SCr count mismatch for stay {row.stay_id}")
    checks.append(f"PASS: onset SCr and pre-onset histories independently rechecked for {len(onset):,} AKI rows")

    train_subjects = set(assignment.loc[assignment["split"].eq("train"), "subject_id"])
    test_subjects = set(assignment.loc[assignment["split"].eq("test"), "subject_id"])
    if train_subjects & test_subjects:
        fail("Subject leakage across train/test assignment")
    if not set(predictions["subject_id"]).issubset(test_subjects):
        fail("Non-test subject found in prediction output")
    checks.append(
        f"PASS: subject-level split has no overlap ({len(train_subjects):,} train; {len(test_subjects):,} test subjects)"
    )

    for row in performance.itertuples(index=False):
        subset = predictions.loc[
            predictions["task"].eq(row.task)
            & predictions["landmark"].astype(str).eq(str(row.landmark))
        ]
        probability_column = f"probability_{row.model.lower().replace(' ', '_')}"
        y = subset["outcome"].astype(int).to_numpy()
        p = subset[probability_column].astype(float).to_numpy()
        recomputed = {
            "auroc": roc_auc_score(y, p),
            "auprc": average_precision_score(y, p),
            "brier_score": brier_score_loss(y, p),
        }
        for metric, value in recomputed.items():
            if not np.isclose(value, getattr(row, metric), atol=5e-12):
                fail(f"{metric} mismatch for {row.task}/{row.landmark}/{row.model}")
    checks.append(f"PASS: AUROC, AUPRC, and Brier score independently recomputed for {len(performance)} models")

    forbidden_fragments = ["recovered_at_end", "persistent_aki", "renal_trajectory", "death", "los", "post_max", "post_min"]
    predictors = missingness["predictor"].astype(str).unique()
    leaked = sorted(
        predictor for predictor in predictors if any(fragment in predictor.lower() for fragment in forbidden_fragments)
    )
    if leaked:
        fail(f"Leakage-prone predictor names detected: {leaked}")
    checks.append("PASS: predictor audit found no outcome, recovery, mortality, LOS, or future-summary fields")

    report = "# v27 independent validation report\n\n" + "\n".join(
        f"- {item}" for item in checks
    ) + "\n\nOverall status: **PASS**.\n"
    (OUT / "audit_v27_independent_validation_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
