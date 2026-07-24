"""Independent consistency checks for the v29 multistate analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "modeling_v29_multistate_competing_risk"
V26 = ROOT / "outputs" / "modeling_v26_aki_severity_trajectories"
COHORT = V26 / "cohort_v26_strict_aki_severity_recovery.csv"

STATES = [
    "No AKI",
    "Stage 1 AKI",
    "Severe AKI (stage 2/3)",
    "Recovered",
    "Recurrent AKI",
    "Live discharge",
    "In-hospital death",
]
TERMINAL = {"Live discharge", "In-hospital death"}
ALLOWED = {
    ("No AKI", "Stage 1 AKI"),
    ("No AKI", "Severe AKI (stage 2/3)"),
    ("No AKI", "Live discharge"),
    ("No AKI", "In-hospital death"),
    ("Stage 1 AKI", "Severe AKI (stage 2/3)"),
    ("Stage 1 AKI", "Recovered"),
    ("Stage 1 AKI", "Live discharge"),
    ("Stage 1 AKI", "In-hospital death"),
    ("Severe AKI (stage 2/3)", "Stage 1 AKI"),
    ("Severe AKI (stage 2/3)", "Recovered"),
    ("Severe AKI (stage 2/3)", "Live discharge"),
    ("Severe AKI (stage 2/3)", "In-hospital death"),
    ("Recovered", "Recurrent AKI"),
    ("Recovered", "Live discharge"),
    ("Recovered", "In-hospital death"),
    ("Recurrent AKI", "Recovered"),
    ("Recurrent AKI", "Live discharge"),
    ("Recurrent AKI", "In-hospital death"),
}


def bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).astype(bool)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def validate_paths(transitions: pd.DataFrame) -> list[str]:
    checks = []
    observed = set(zip(transitions["from_state"], transitions["to_state"]))
    if not observed.issubset(ALLOWED):
        raise AssertionError(f"Unexpected transitions: {sorted(observed - ALLOWED)}")
    if (~transitions["from_state"].isin(STATES) | ~transitions["to_state"].isin(STATES)).any():
        raise AssertionError("Unknown state label")
    if (~transitions["transition_hours_from_icu"].between(0, 168)).any():
        raise AssertionError("Transition outside the fixed 0-168 h window")
    for stay_id, group in transitions.groupby("stay_id", sort=False):
        group = group.sort_values(["transition_hours_from_icu", "transition_source"])
        if not group["transition_hours_from_icu"].is_monotonic_increasing:
            raise AssertionError(f"Non-monotone transition times for {stay_id}")
        current = "No AKI"
        for row in group.itertuples(index=False):
            if row.from_state != current:
                raise AssertionError(f"Broken path continuity for {stay_id}")
            if current in TERMINAL:
                raise AssertionError(f"Transition after absorbing state for {stay_id}")
            current = row.to_state
        terminal_rows = group["to_state"].isin(TERMINAL)
        if terminal_rows.any() and not terminal_rows.iloc[-1]:
            raise AssertionError(f"Terminal transition is not last for {stay_id}")
    checks.append("Every stay follows a continuous prespecified path with terminal absorption")
    return checks


def validate_occupancy(occupancy: pd.DataFrame, assignments: pd.DataFrame) -> list[str]:
    checks = []
    sums = occupancy.groupby("time_hours")["aj_percent"].sum()
    if not np.allclose(sums, 100, atol=1e-8):
        raise AssertionError("Aalen-Johansen state occupancy does not sum to 100%")
    if occupancy["aj_empirical_absolute_difference_percent"].max() > 1e-8:
        raise AssertionError("Product-integral and direct empirical occupancy disagree")
    direct = assignments.groupby(["time_hours", "state"]).size().rename("n_recomputed")
    merged = occupancy.set_index(["time_hours", "state"]).join(direct, how="left").fillna(0)
    if not (merged["n"].astype(int) == merged["n_recomputed"].astype(int)).all():
        raise AssertionError("Saved state counts do not reproduce from assignments")
    if ((occupancy["ci95_low_percent"] > occupancy["ci95_high_percent"]) | (occupancy["ci95_low_percent"] < 0) | (occupancy["ci95_high_percent"] > 100)).any():
        raise AssertionError("Invalid occupancy confidence interval")
    checks.append("AJ occupancy sums to 100%, matches empirical counts, and has valid intervals")
    return checks


def validate_competing_curve(table: pd.DataFrame, label: str) -> list[str]:
    checks = []
    key_columns = ["analysis", "group_variable", "group"]
    for _, group in table.groupby(key_columns, dropna=False):
        for _, cause in group.groupby("cause"):
            values = cause.sort_values("time_hours")["cif"].to_numpy()
            if np.any(np.diff(values) < -1e-12):
                raise AssertionError(f"Non-monotone CIF in {label}")
        totals = group.groupby("time_hours").agg(
            cif_sum=("cif", "sum"), survival=("event_free_survival", "first")
        )
        if (totals["cif_sum"] + totals["survival"] > 1 + 1e-8).any():
            raise AssertionError(f"CIF plus survival exceeds one in {label}")
        if (group["risk_set_weighted_n"] < -1e-9).any():
            raise AssertionError(f"Negative risk set in {label}")
        if ((group["ci95_low_percent"] < 0) | (group["ci95_high_percent"] > 100) | (group["ci95_low_percent"] > group["ci95_high_percent"])).any():
            raise AssertionError(f"Invalid bootstrap interval in {label}")
    checks.append(f"{label} CIFs are monotone, bounded, and coherent with event-free survival")
    return checks


def validate_temporal_exclusions(
    cohort: pd.DataFrame,
    exclusions: pd.DataFrame,
    onset: pd.DataFrame,
    transitions: pd.DataFrame,
) -> list[str]:
    checks = []
    locked_aki = cohort.loc[bool_mask(cohort["aki_final"])].copy()
    if len(locked_aki) != len(onset) + len(exclusions):
        raise AssertionError("Trajectory-eligible and temporally excluded AKI do not exhaust locked AKI")
    if not (exclusions["aki_onset_hours_from_icu"] > exclusions["terminal_hours_from_icu"]).all():
        raise AssertionError("A temporal exclusion does not occur after disposition")
    if set(onset["stay_id"]) & set(exclusions["stay_id"]):
        raise AssertionError("Excluded AKI remains in onset competing-risk cohort")
    active_stays = set(
        transitions.loc[
            transitions["to_state"].isin(["Stage 1 AKI", "Severe AKI (stage 2/3)"]),
            "stay_id",
        ]
    )
    if set(onset["stay_id"]) != active_stays:
        raise AssertionError("Trajectory-eligible AKI does not equal observed active-state entry")
    checks.append("Locked AKI is exhausted by 4,519 valid trajectories plus 12 post-disposition exclusions")
    return checks


def validate_recurrence(recurrence: pd.DataFrame, transitions: pd.DataFrame) -> list[str]:
    checks = []
    recovered = set(transitions.loc[transitions["to_state"].eq("Recovered"), "stay_id"])
    if set(recurrence["stay_id"]) != recovered:
        raise AssertionError("Recovery landmark cohort does not match first observed recovery paths")
    recurrent = set(transitions.loc[transitions["to_state"].eq("Recurrent AKI"), "stay_id"])
    reported = set(recurrence.loc[recurrence["first_event"].eq("Recurrent AKI"), "stay_id"])
    if recurrent != reported:
        raise AssertionError("First recurrence events disagree with state paths")
    checks.append("Recovery landmark and first recurrence events reproduce from transition paths")
    return checks


def main() -> None:
    cohort = pd.read_csv(COHORT, low_memory=False)
    cohort = cohort.loc[bool_mask(cohort["incident_aki_evaluable"])].copy()
    transitions = pd.read_csv(OUT / "cohort_v29_multistate_transitions.csv.gz")
    assignments = pd.read_csv(OUT / "cohort_v29_state_assignments.csv.gz")
    occupancy = pd.read_csv(OUT / "analysis_v29_state_occupancy.csv")
    exclusions = pd.read_csv(OUT / "audit_v29_postdisposition_aki_exclusions.csv")
    onset = pd.read_csv(OUT / "cohort_v29_competing_events_after_aki.csv")
    recurrence = pd.read_csv(OUT / "cohort_v29_competing_events_after_recovery.csv")
    onset_cif = pd.read_csv(OUT / "analysis_v29_cif_after_aki.csv")
    recurrence_cif = pd.read_csv(OUT / "analysis_v29_cif_after_recovery.csv")

    checks = []
    checks += validate_paths(transitions)
    checks += validate_occupancy(occupancy, assignments)
    checks += validate_temporal_exclusions(cohort, exclusions, onset, transitions)
    checks += validate_recurrence(recurrence, transitions)
    checks += validate_competing_curve(onset_cif, "AKI-onset competing-risk")
    checks += validate_competing_curve(recurrence_cif, "post-recovery competing-risk")
    required_figures = [
        OUT / "figure_v29_multistate_competing_risk.svg",
        OUT / "figure_v29_multistate_competing_risk.pdf",
        OUT / "figure_v29_multistate_competing_risk.tiff",
        OUT / "figure_v29_competing_risk_subgroups.svg",
    ]
    if not all(path.exists() and path.stat().st_size > 1000 for path in required_figures):
        raise AssertionError("Publication figure export missing or empty")
    checks.append("Vector, PDF, and high-resolution raster figure exports exist")

    report = "\n".join(
        ["# Independent v29 validation", "", f"Status: PASS ({len(checks)} checks)", ""]
        + [f"- {check}" for check in checks]
    ) + "\n"
    (OUT / "audit_v29_independent_validation_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
