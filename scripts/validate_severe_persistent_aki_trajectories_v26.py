"""Independent consistency checks for the v26 AKI trajectory outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "modeling_v26_aki_severity_trajectories"
COHORT = OUT / "cohort_v26_strict_aki_severity_recovery.csv"
STATES = OUT / "creatinine_measurement_states_v26.csv.gz"
SOURCE = ROOT / "outputs" / "finalized_v3_1" / "cohort_v3_1_strict_main_evaluable.csv"


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> None:
    source = pd.read_csv(SOURCE, low_memory=False)
    cohort = pd.read_csv(
        COHORT,
        low_memory=False,
        parse_dates=[
            "intime",
            "dischtime",
            "aki_onset_time_final",
            "first_recovery_time",
            "end_observation_time",
            "end_scr_time",
            "rrt_first_time",
        ],
    )
    states = pd.read_csv(STATES, low_memory=False, parse_dates=["charttime"])
    checks: list[str] = []

    if len(source) != len(cohort) or set(source["stay_id"]) != set(cohort["stay_id"]):
        fail("Patient-level output does not retain the exact source cohort")
    checks.append(f"PASS: exact source-cohort coverage ({len(cohort):,}/{len(source):,})")

    source_stage = source["aki_stage_final"].value_counts().sort_index()
    output_stage = cohort["aki_stage_final"].value_counts().sort_index()
    if not source_stage.equals(output_stage):
        fail("Locked v3 stage distribution changed")
    checks.append(f"PASS: original AKI/stage counts retained (AKI n={int(cohort['aki_final'].sum()):,})")

    # Recompute the ratio from the original stored baseline rather than the
    # rounded CSV ratio.  This intentionally reproduces the locked v3 floating-
    # point boundary behavior for exact nominal 3.0-fold values (for example
    # 1.2/0.4 may be represented as 2.9999999999999996).
    baseline = cohort[["stay_id", "baseline_scr_final"]]
    states = states.merge(baseline, on="stay_id", how="left", validate="many_to_one")
    ratio = (
        states["scr_mg_dl"].to_numpy(float)
        / states["baseline_scr_final"].to_numpy(float)
    )
    scr = states["scr_mg_dl"].to_numpy(float)
    absolute = states["aki_absolute_0_3_within_prior_48h"].astype(bool).to_numpy()
    expected = np.select(
        [
            (ratio >= 3.0) | (scr >= 4.0),
            ratio >= 2.0,
            (ratio >= 1.5) | absolute,
        ],
        [3, 2, 1],
        default=0,
    )
    mismatch = int(np.sum(expected != states["aki_stage_at_measurement"].to_numpy(int)))
    if mismatch:
        fail(f"Measurement-stage recomputation mismatch: {mismatch}")
    checks.append(f"PASS: independently recomputed all {len(states):,} measurement-level stages")

    severe_expected = cohort["aki_stage_final"].ge(2)
    if not severe_expected.equals(cohort["severe_aki_scr_stage2_3"].astype(bool)):
        fail("Severe-AKI flag does not equal original SCr stage 2/3")
    checks.append(f"PASS: severe SCr AKI flag recomputed (n={int(severe_expected.sum()):,})")

    state_groups = {stay: group.sort_values("charttime") for stay, group in states.groupby("stay_id")}
    checked_aki = 0
    end_evaluable = 0
    for row in cohort.loc[cohort["aki_final"]].itertuples(index=False):
        episode = state_groups[row.stay_id]
        episode = episode.loc[episode["charttime"].ge(row.aki_onset_time_final)]
        recovered = episode.loc[~episode["aki_active_by_kdigo_scr"].astype(bool)]
        first = recovered["charttime"].min() if len(recovered) else pd.NaT
        expected_rapid = bool(
            pd.notna(first)
            and (first - row.aki_onset_time_final).total_seconds() / 3600 <= 48 + 1e-12
        )
        if expected_rapid != bool(row.rapid_reversal_within_48h):
            fail(f"Rapid-reversal mismatch for stay {row.stay_id}")
        active_after = episode.loc[
            episode["hours_from_aki_onset"].ge(48)
            & episode["aki_active_by_kdigo_scr"].astype(bool)
        ]
        if row.aki_duration_class == "Persistent AKI" and (expected_rapid or active_after.empty):
            fail(f"Persistent-AKI evidence mismatch for stay {row.stay_id}")

        end_window = episode.loc[
            episode["charttime"].le(row.end_observation_time)
            & episode["charttime"].ge(row.end_observation_time - pd.Timedelta(hours=24))
        ]
        expected_end_eval = not end_window.empty
        if expected_end_eval != bool(row.end_recovery_evaluable):
            fail(f"End-window evaluability mismatch for stay {row.stay_id}")
        if expected_end_eval:
            end_evaluable += 1
            last = end_window.iloc[-1]
            if abs((last["charttime"] - row.end_scr_time).total_seconds()) > 1e-6:
                fail(f"End SCr time mismatch for stay {row.stay_id}")
            if int(last["aki_stage_at_measurement"]) != int(row.end_aki_stage_observed):
                fail(f"End SCr stage mismatch for stay {row.stay_id}")
        checked_aki += 1
    checks.append(f"PASS: rapid reversal and >48 h persistence rechecked for {checked_aki:,} AKI stays")
    checks.append(f"PASS: end-window recovery rechecked for {end_evaluable:,} evaluable AKI stays")

    rrt = cohort.loc[cohort["rrt_within_7d"].astype(bool)]
    if ((rrt["rrt_first_time"] - rrt["intime"]).dt.total_seconds() / 3600 > 168).any():
        fail("RRT event beyond day 7")
    if (rrt["aki_stage_scr_or_rrt_7d"] != 3).any():
        fail("RRT sensitivity stage was not upgraded to stage 3")
    checks.append(f"PASS: RRT timing and sensitivity staging rechecked (n={len(rrt):,})")

    summary = pd.read_csv(OUT / "audit_v26_severity_summary.csv")
    reported_severe = int(
        summary.loc[summary["category"].eq("Severe AKI (stage 2/3)"), "n"].iloc[0]
    )
    if reported_severe != int(severe_expected.sum()):
        fail("Severity audit table does not match patient-level data")
    checks.append("PASS: audit table counts reconcile with the patient-level phenotype file")

    report = "# v26 independent validation report\n\n" + "\n".join(
        f"- {item}" for item in checks
    ) + "\n\nOverall status: **PASS**.\n"
    (OUT / "audit_v26_independent_validation_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
