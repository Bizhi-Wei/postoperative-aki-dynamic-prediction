"""Non-parametric multistate and competing-risk analysis of postoperative AKI.

The fixed multistate clock starts at ICU admission and ends at 168 hours.
Serum-creatinine states update only when SCr is observed. In-hospital death and
live hospital discharge are absorbing states. Aalen--Johansen state occupancy
is supplemented by cause-specific cumulative incidence from AKI onset and
from first observed recovery. Confidence intervals use subject-cluster
bootstrap resampling so repeated admissions remain within the same cluster.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V26 = ROOT / "outputs" / "modeling_v26_aki_severity_trajectories"
COHORT_FILE = V26 / "cohort_v26_strict_aki_severity_recovery.csv"
STATES_FILE = V26 / "creatinine_measurement_states_v26.csv.gz"
OUT = ROOT / "outputs" / "modeling_v29_multistate_competing_risk"

HORIZON_HOURS = 168.0
BOOTSTRAPS = 500
RANDOM_STATE = 20260729
GRID_HOURS = np.array([0, 6, 12, 24, 48, 72, 96, 120, 144, 168], dtype=float)
CIF_GRID_HOURS = np.array([0, 6, 12, 24, 48, 72, 96, 120, 144], dtype=float)

STATE_ORDER = [
    "No AKI",
    "Stage 1 AKI",
    "Severe AKI (stage 2/3)",
    "Recovered",
    "Recurrent AKI",
    "Live discharge",
    "In-hospital death",
]
TRANSIENT_STATES = set(STATE_ORDER[:5])
STATE_COLORS = {
    "No AKI": "#D0D5DD",
    "Stage 1 AKI": "#F2B66D",
    "Severe AKI (stage 2/3)": "#B42318",
    "Recovered": "#72B7B2",
    "Recurrent AKI": "#7B61A8",
    "Live discharge": "#4C78A8",
    "In-hospital death": "#344054",
}
CAUSE_COLORS = {
    "Observed recovery": "#218C7A",
    "Severe AKI onset/progression": "#B42318",
    "Recurrent AKI": "#7B61A8",
    "Live discharge": "#4C78A8",
    "In-hospital death": "#344054",
}


def bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).astype(bool)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 5.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in {
        "png": {"dpi": 300},
        "pdf": {},
        "svg": {},
        "tiff": {"dpi": 600},
    }.items():
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(COHORT_FILE, low_memory=False)
    states = pd.read_csv(STATES_FILE, low_memory=False)
    for column in ["intime", "dischtime", "aki_onset_time_final"]:
        cohort[column] = pd.to_datetime(cohort[column], errors="coerce")
    cohort = cohort.loc[bool_mask(cohort["incident_aki_evaluable"])].copy()
    states = states.loc[states["stay_id"].isin(cohort["stay_id"])].copy()
    states["hours_from_icu"] = pd.to_numeric(states["hours_from_icu"], errors="coerce")
    states["aki_stage_at_measurement"] = pd.to_numeric(
        states["aki_stage_at_measurement"], errors="coerce"
    )
    states = states.loc[
        states["hours_from_icu"].between(0, HORIZON_HOURS, inclusive="both")
        & states["aki_stage_at_measurement"].notna()
    ].copy()
    if cohort["stay_id"].duplicated().any():
        raise ValueError("Duplicate stay_id in v26 evaluable cohort")
    if not set(states["stay_id"]).issubset(set(cohort["stay_id"])):
        raise ValueError("Measurement state contains a stay outside the analysis cohort")
    return cohort, states


def terminal_table(cohort: pd.DataFrame) -> pd.DataFrame:
    result = cohort[["subject_id", "hadm_id", "stay_id", "intime", "dischtime"]].copy()
    result["terminal_hours"] = (
        (result["dischtime"] - result["intime"]).dt.total_seconds() / 3600
    )
    result["terminal_hours"] = result["terminal_hours"].where(
        result["terminal_hours"].between(0, HORIZON_HOURS, inclusive="both")
    )
    death = bool_mask(cohort["hosp_death"]) | bool_mask(cohort["hospital_expire_flag"])
    result["terminal_state"] = np.where(death, "In-hospital death", "Live discharge")
    result.loc[result["terminal_hours"].isna(), "terminal_state"] = pd.NA
    return result


def state_after_measurement(
    current: str, stage: int, ever_aki: bool, recovered_once: bool
) -> tuple[str, bool, bool]:
    if not ever_aki:
        if stage <= 0:
            return current, False, False
        return ("Stage 1 AKI" if stage == 1 else "Severe AKI (stage 2/3)"), True, False
    if recovered_once:
        if stage <= 0:
            return "Recovered", True, True
        return "Recurrent AKI", True, True
    if stage <= 0:
        return "Recovered", True, True
    return ("Stage 1 AKI" if stage == 1 else "Severe AKI (stage 2/3)"), True, False


def build_transition_process(
    cohort: pd.DataFrame, states: pd.DataFrame, terminals: pd.DataFrame
) -> pd.DataFrame:
    terminal_index = terminals.set_index("stay_id")
    measurements = (
        states.groupby(["stay_id", "hours_from_icu"], as_index=False)["aki_stage_at_measurement"]
        .max()
        .sort_values(["stay_id", "hours_from_icu"])
    )
    by_stay = {stay: group for stay, group in measurements.groupby("stay_id", sort=False)}
    rows: list[dict[str, object]] = []
    for patient in cohort[["subject_id", "hadm_id", "stay_id"]].itertuples(index=False):
        terminal_hours = terminal_index.at[patient.stay_id, "terminal_hours"]
        terminal_state = terminal_index.at[patient.stay_id, "terminal_state"]
        current = "No AKI"
        ever_aki = False
        recovered_once = False
        group = by_stay.get(patient.stay_id)
        if group is not None:
            for measurement in group.itertuples(index=False):
                time = float(measurement.hours_from_icu)
                if pd.notna(terminal_hours) and time >= float(terminal_hours):
                    continue
                new_state, ever_aki, recovered_once = state_after_measurement(
                    current,
                    int(measurement.aki_stage_at_measurement),
                    ever_aki,
                    recovered_once,
                )
                if new_state != current:
                    rows.append(
                        {
                            "subject_id": patient.subject_id,
                            "hadm_id": patient.hadm_id,
                            "stay_id": patient.stay_id,
                            "transition_hours_from_icu": time,
                            "from_state": current,
                            "to_state": new_state,
                            "transition_source": "serum creatinine",
                        }
                    )
                    current = new_state
        if pd.notna(terminal_hours):
            rows.append(
                {
                    "subject_id": patient.subject_id,
                    "hadm_id": patient.hadm_id,
                    "stay_id": patient.stay_id,
                    "transition_hours_from_icu": float(terminal_hours),
                    "from_state": current,
                    "to_state": str(terminal_state),
                    "transition_source": "hospital disposition",
                }
            )
    transitions = pd.DataFrame(rows)
    transitions = transitions.sort_values(
        ["stay_id", "transition_hours_from_icu", "transition_source"]
    ).reset_index(drop=True)
    return transitions


def empirical_assignments(
    cohort: pd.DataFrame, transitions: pd.DataFrame, grid: np.ndarray
) -> pd.DataFrame:
    transition_groups = {stay: group for stay, group in transitions.groupby("stay_id", sort=False)}
    rows: list[dict[str, object]] = []
    for patient in cohort[["subject_id", "hadm_id", "stay_id"]].itertuples(index=False):
        group = transition_groups.get(patient.stay_id)
        times = np.array([], dtype=float) if group is None else group["transition_hours_from_icu"].to_numpy(float)
        states = np.array([], dtype=object) if group is None else group["to_state"].to_numpy(object)
        for time in grid:
            index = np.searchsorted(times, time, side="right") - 1
            state = "No AKI" if index < 0 else str(states[index])
            rows.append(
                {
                    "subject_id": patient.subject_id,
                    "hadm_id": patient.hadm_id,
                    "stay_id": patient.stay_id,
                    "time_hours": float(time),
                    "state": state,
                }
            )
    return pd.DataFrame(rows)


def aalen_johansen_occupancy(
    transitions: pd.DataFrame, n: int, grid: np.ndarray
) -> pd.DataFrame:
    state_index = {state: i for i, state in enumerate(STATE_ORDER)}
    probability = np.zeros(len(STATE_ORDER), dtype=float)
    probability[state_index["No AKI"]] = 1.0
    risk_counts = np.zeros(len(STATE_ORDER), dtype=float)
    risk_counts[state_index["No AKI"]] = n
    groups = list(transitions.groupby("transition_hours_from_icu", sort=True))
    pointer = 0
    rows = []
    for time in grid:
        while pointer < len(groups) and float(groups[pointer][0]) <= float(time):
            _, events = groups[pointer]
            counts = (
                events.groupby(["from_state", "to_state"]).size().rename("n").reset_index()
            )
            increment = np.zeros((len(STATE_ORDER), len(STATE_ORDER)), dtype=float)
            flows = np.zeros_like(risk_counts)
            arrivals = np.zeros_like(risk_counts)
            for event in counts.itertuples(index=False):
                j, k = state_index[event.from_state], state_index[event.to_state]
                if risk_counts[j] <= 0:
                    raise AssertionError(f"Empty risk state for {event.from_state} transition")
                increment[j, k] += event.n / risk_counts[j]
                increment[j, j] -= event.n / risk_counts[j]
                flows[j] += event.n
                arrivals[k] += event.n
            probability = probability @ (np.eye(len(STATE_ORDER)) + increment)
            risk_counts = risk_counts - flows + arrivals
            pointer += 1
        for state, value in zip(STATE_ORDER, probability):
            rows.append(
                {
                    "time_hours": float(time),
                    "state": state,
                    "aj_probability": float(value),
                    "aj_percent": 100 * float(value),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_occupancy(
    assignments: pd.DataFrame, grid: np.ndarray, draws: int = BOOTSTRAPS
) -> pd.DataFrame:
    subject_state = (
        assignments.groupby(["subject_id", "time_hours", "state"])
        .size()
        .unstack(["time_hours", "state"], fill_value=0)
    )
    full_columns = pd.MultiIndex.from_product([grid.astype(float), STATE_ORDER])
    subject_state = subject_state.reindex(columns=full_columns, fill_value=0)
    matrix = subject_state.to_numpy(float)
    n_subjects = len(subject_state)
    rng = np.random.default_rng(RANDOM_STATE)
    estimates = np.empty((draws, matrix.shape[1]), dtype=float)
    for draw in range(draws):
        sampled = rng.integers(0, n_subjects, size=n_subjects)
        weights = np.bincount(sampled, minlength=n_subjects)
        totals = weights @ matrix
        denominator = totals.reshape(len(grid), len(STATE_ORDER)).sum(axis=1)
        estimates[draw] = (
            totals.reshape(len(grid), len(STATE_ORDER)) / denominator[:, None]
        ).reshape(-1)
    low = np.quantile(estimates, 0.025, axis=0)
    high = np.quantile(estimates, 0.975, axis=0)
    rows = []
    index = 0
    for time in grid:
        for state in STATE_ORDER:
            rows.append(
                {
                    "time_hours": float(time),
                    "state": state,
                    "ci95_low_percent": 100 * low[index],
                    "ci95_high_percent": 100 * high[index],
                    "bootstrap_n": draws,
                }
            )
            index += 1
    return pd.DataFrame(rows)


def first_measurement_times(states: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {stay: group.sort_values("hours_from_icu") for stay, group in states.groupby("stay_id")}


def competing_dataset_after_aki(
    cohort: pd.DataFrame, transitions: pd.DataFrame, terminals: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    terminal_index = terminals.set_index("stay_id")
    transition_groups = {
        stay: group.sort_values("transition_hours_from_icu")
        for stay, group in transitions.groupby("stay_id")
    }
    rows = []
    exclusions = []
    aki = cohort.loc[bool_mask(cohort["aki_final"])].copy()
    for patient in aki.itertuples(index=False):
        onset = float(patient.aki_onset_hours_final)
        censor = max(0.0, HORIZON_HOURS - onset)
        terminal_hours = terminal_index.at[patient.stay_id, "terminal_hours"]
        terminal_state = terminal_index.at[patient.stay_id, "terminal_state"]
        if pd.notna(terminal_hours) and onset > float(terminal_hours) + 1e-9:
            exclusions.append(
                {
                    "subject_id": patient.subject_id,
                    "hadm_id": patient.hadm_id,
                    "stay_id": patient.stay_id,
                    "aki_onset_hours_from_icu": onset,
                    "terminal_hours_from_icu": float(terminal_hours),
                    "terminal_state": terminal_state,
                    "exclusion_reason": "locked AKI onset after recorded hospital disposition",
                }
            )
            continue
        group = transition_groups.get(patient.stay_id)
        candidates: list[tuple[float, int, str]] = []
        if group is not None:
            after = group.loc[group["transition_hours_from_icu"].ge(onset - 1e-9)].copy()
            severe = after.loc[
                after["to_state"].eq("Severe AKI (stage 2/3)"),
                "transition_hours_from_icu",
            ]
            recovery = after.loc[
                after["transition_hours_from_icu"].gt(onset + 1e-9)
                & after["to_state"].eq("Recovered"),
                "transition_hours_from_icu",
            ]
            if len(severe):
                candidates.append(
                    (max(0.0, float(severe.min()) - onset), 1, "Severe AKI onset/progression")
                )
            if len(recovery):
                candidates.append((float(recovery.min()) - onset, 2, "Observed recovery"))
        if pd.notna(terminal_hours) and float(terminal_hours) >= onset:
            cause = str(terminal_state)
            priority = 0 if cause == "In-hospital death" else 3
            candidates.append((float(terminal_hours) - onset, priority, cause))
        candidates = [candidate for candidate in candidates if candidate[0] <= censor + 1e-9]
        if candidates:
            event_time, _, event = sorted(candidates, key=lambda x: (x[0], x[1]))[0]
        else:
            event_time, event = censor, "Administrative censoring"
        rows.append(
            {
                "subject_id": patient.subject_id,
                "hadm_id": patient.hadm_id,
                "stay_id": patient.stay_id,
                "aki_onset_hours_from_icu": onset,
                "observed_time_hours": float(event_time),
                "first_event": event,
                "ckd": bool(patient.ckd),
                "cardiac_surgery": bool(patient.cardiac_surgery),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(exclusions)


def recurrence_dataset(
    cohort: pd.DataFrame, transitions: pd.DataFrame, terminals: pd.DataFrame
) -> pd.DataFrame:
    terminal_index = terminals.set_index("stay_id")
    transition_groups = {
        stay: group.sort_values("transition_hours_from_icu")
        for stay, group in transitions.groupby("stay_id")
    }
    rows = []
    aki = cohort.loc[bool_mask(cohort["aki_final"])].copy()
    for patient in aki.itertuples(index=False):
        onset = float(patient.aki_onset_hours_final)
        terminal_hours = terminal_index.at[patient.stay_id, "terminal_hours"]
        terminal_state = terminal_index.at[patient.stay_id, "terminal_state"]
        if pd.notna(terminal_hours) and onset > float(terminal_hours) + 1e-9:
            continue
        group = transition_groups.get(patient.stay_id)
        if group is None:
            continue
        recovery = group.loc[
            group["transition_hours_from_icu"].gt(onset + 1e-9)
            & group["to_state"].eq("Recovered"),
            "transition_hours_from_icu",
        ]
        if not len(recovery):
            continue
        recovery_time = float(recovery.min())
        censor = max(0.0, HORIZON_HOURS - recovery_time)
        candidates: list[tuple[float, int, str]] = []
        recurrence = group.loc[
            group["transition_hours_from_icu"].gt(recovery_time + 1e-9)
            & group["to_state"].eq("Recurrent AKI"),
            "transition_hours_from_icu",
        ]
        if len(recurrence):
            candidates.append((float(recurrence.min()) - recovery_time, 1, "Recurrent AKI"))
        if pd.notna(terminal_hours) and float(terminal_hours) >= recovery_time:
            cause = str(terminal_state)
            priority = 0 if cause == "In-hospital death" else 2
            candidates.append((float(terminal_hours) - recovery_time, priority, cause))
        candidates = [candidate for candidate in candidates if candidate[0] <= censor + 1e-9]
        if candidates:
            event_time, _, event = sorted(candidates, key=lambda x: (x[0], x[1]))[0]
        else:
            event_time, event = censor, "Administrative censoring"
        rows.append(
            {
                "subject_id": patient.subject_id,
                "hadm_id": patient.hadm_id,
                "stay_id": patient.stay_id,
                "first_recovery_hours_from_icu": recovery_time,
                "observed_time_hours": float(event_time),
                "first_event": event,
                "ckd": bool(patient.ckd),
                "cardiac_surgery": bool(patient.cardiac_surgery),
            }
        )
    return pd.DataFrame(rows)


def cif_at_grid(
    data: pd.DataFrame,
    causes: list[str],
    grid: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = data["observed_time_hours"].to_numpy(float)
    events = data["first_event"].astype(str).to_numpy()
    weights = np.ones(len(data), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(times, kind="mergesort")
    times, events, weights = times[order], events[order], weights[order]
    unique_times, starts = np.unique(times, return_index=True)
    ends = np.r_[starts[1:], len(times)]
    risk = float(weights.sum())
    survival = 1.0
    cif = np.zeros(len(causes), dtype=float)
    cif_grid = np.zeros((len(grid), len(causes)), dtype=float)
    survival_grid = np.zeros(len(grid), dtype=float)
    risk_grid = np.zeros(len(grid), dtype=float)
    pointer = 0
    for grid_index, grid_time in enumerate(grid):
        while pointer < len(unique_times) and unique_times[pointer] <= grid_time + 1e-12:
            block = slice(starts[pointer], ends[pointer])
            event_block, weight_block = events[block], weights[block]
            if risk > 0:
                event_mask = event_block != "Administrative censoring"
                total_events = float(weight_block[event_mask].sum())
                for cause_index, cause in enumerate(causes):
                    cause_events = float(weight_block[event_block == cause].sum())
                    cif[cause_index] += survival * cause_events / risk
                survival *= max(0.0, 1.0 - total_events / risk)
            risk -= float(weight_block.sum())
            pointer += 1
        cif_grid[grid_index] = cif
        survival_grid[grid_index] = survival
        risk_grid[grid_index] = max(risk, 0.0)
    return cif_grid, survival_grid, risk_grid


def clustered_cif(
    data: pd.DataFrame,
    causes: list[str],
    grid: np.ndarray,
    analysis: str,
    group_variable: str,
    group: str,
    draws: int = BOOTSTRAPS,
) -> pd.DataFrame:
    point, survival, risk = cif_at_grid(data, causes, grid)
    subjects, codes = np.unique(data["subject_id"].astype(str), return_inverse=True)
    rng = np.random.default_rng(RANDOM_STATE + len(data) + len(group))
    boot = np.empty((draws, len(grid), len(causes)), dtype=float)
    for draw in range(draws):
        sampled = rng.integers(0, len(subjects), size=len(subjects))
        subject_weights = np.bincount(sampled, minlength=len(subjects))
        boot[draw], _, _ = cif_at_grid(data, causes, grid, subject_weights[codes])
    low = np.quantile(boot, 0.025, axis=0)
    high = np.quantile(boot, 0.975, axis=0)
    rows = []
    for time_index, time in enumerate(grid):
        for cause_index, cause in enumerate(causes):
            rows.append(
                {
                    "analysis": analysis,
                    "group_variable": group_variable,
                    "group": group,
                    "n": len(data),
                    "time_hours": float(time),
                    "cause": cause,
                    "cif": point[time_index, cause_index],
                    "cif_percent": 100 * point[time_index, cause_index],
                    "ci95_low_percent": 100 * low[time_index, cause_index],
                    "ci95_high_percent": 100 * high[time_index, cause_index],
                    "event_free_survival": survival[time_index],
                    "risk_set_weighted_n": risk[time_index],
                    "bootstrap_n": draws,
                }
            )
    return pd.DataFrame(rows)


def all_cif_analyses(
    onset_data: pd.DataFrame, recurrence_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    onset_causes = [
        "Observed recovery",
        "Severe AKI onset/progression",
        "Live discharge",
        "In-hospital death",
    ]
    recurrence_causes = ["Recurrent AKI", "Live discharge", "In-hospital death"]
    onset_blocks = [
        clustered_cif(
            onset_data,
            onset_causes,
            CIF_GRID_HOURS,
            "First event after incident AKI onset",
            "Overall",
            "All incident AKI",
        )
    ]
    recurrence_blocks = [
        clustered_cif(
            recurrence_data,
            recurrence_causes,
            CIF_GRID_HOURS,
            "First event after observed recovery",
            "Overall",
            "All observed recoveries",
        )
    ]
    strata = [
        ("CKD", "ckd", [(False, "No CKD"), (True, "CKD")]),
        (
            "Surgery type",
            "cardiac_surgery",
            [(False, "Non-cardiac surgery"), (True, "Cardiac surgery")],
        ),
    ]
    for variable_label, column, levels in strata:
        for value, label in levels:
            onset_subset = onset_data.loc[bool_mask(onset_data[column]).eq(value)].copy()
            recurrence_subset = recurrence_data.loc[
                bool_mask(recurrence_data[column]).eq(value)
            ].copy()
            onset_blocks.append(
                clustered_cif(
                    onset_subset,
                    onset_causes,
                    CIF_GRID_HOURS,
                    "First event after incident AKI onset",
                    variable_label,
                    label,
                )
            )
            recurrence_blocks.append(
                clustered_cif(
                    recurrence_subset,
                    recurrence_causes,
                    CIF_GRID_HOURS,
                    "First event after observed recovery",
                    variable_label,
                    label,
                )
            )
    return pd.concat(onset_blocks, ignore_index=True), pd.concat(
        recurrence_blocks, ignore_index=True
    )


def transition_summary(transitions: pd.DataFrame) -> pd.DataFrame:
    return (
        transitions.groupby(["from_state", "to_state", "transition_source"])
        .agg(
            transition_n=("stay_id", "size"),
            stay_n=("stay_id", "nunique"),
            subject_n=("subject_id", "nunique"),
            transition_time_median_hours=("transition_hours_from_icu", "median"),
            transition_time_q1_hours=("transition_hours_from_icu", lambda x: x.quantile(0.25)),
            transition_time_q3_hours=("transition_hours_from_icu", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )


def overall_summary(
    cohort: pd.DataFrame,
    transitions: pd.DataFrame,
    onset_data: pd.DataFrame,
    recurrence_data: pd.DataFrame,
    temporal_exclusions: pd.DataFrame,
) -> pd.DataFrame:
    values = [
        ("Analysis cohort", len(cohort), len(cohort)),
        ("Incident SCr-AKI", int(bool_mask(cohort["aki_final"]).sum()), len(cohort)),
        (
            "Trajectory-eligible incident SCr-AKI",
            len(onset_data),
            int(bool_mask(cohort["aki_final"]).sum()),
        ),
        (
            "Locked AKI onset after recorded hospital disposition",
            len(temporal_exclusions),
            int(bool_mask(cohort["aki_final"]).sum()),
        ),
        (
            "Severe SCr-AKI stage 2/3",
            int(cohort["maximum_active_scr_stage_7d"].ge(2).sum()),
            len(cohort),
        ),
        ("Observed recovery after AKI", len(recurrence_data), len(onset_data)),
        (
            "Recurrent AKI after observed recovery",
            int(recurrence_data["first_event"].eq("Recurrent AKI").sum()),
            len(recurrence_data),
        ),
        (
            "Live discharge within 7 days",
            int(transitions["to_state"].eq("Live discharge").sum()),
            len(cohort),
        ),
        (
            "In-hospital death within 7 days",
            int(transitions["to_state"].eq("In-hospital death").sum()),
            len(cohort),
        ),
    ]
    return pd.DataFrame(
        [
            {
                "measure": label,
                "n": numerator,
                "denominator_n": denominator,
                "percent": 100 * numerator / denominator if denominator else np.nan,
            }
            for label, numerator, denominator in values
        ]
    )


def plot_cif_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    causes: list[str],
    xlabel: str,
    maximum_days: float,
) -> None:
    overall = data.loc[data["group_variable"].eq("Overall")]
    for cause in causes:
        subset = overall.loc[overall["cause"].eq(cause)].sort_values("time_hours")
        ax.plot(
            subset["time_hours"] / 24,
            subset["cif_percent"],
            color=CAUSE_COLORS[cause],
            linewidth=1.45,
            label=cause,
        )
        ax.fill_between(
            subset["time_hours"] / 24,
            subset["ci95_low_percent"],
            subset["ci95_high_percent"],
            color=CAUSE_COLORS[cause],
            alpha=0.12,
            linewidth=0,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative incidence (%)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlim(0, maximum_days)
    ax.set_ylim(bottom=0)


def make_figures(
    occupancy: pd.DataFrame,
    onset_cif: pd.DataFrame,
    recurrence_cif: pd.DataFrame,
) -> None:
    setup_style()
    fig = plt.figure(figsize=(7.2, 4.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1])
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    pivot = occupancy.pivot(index="time_hours", columns="state", values="aj_percent")
    x = pivot.index.to_numpy(float) / 24
    ax_a.stackplot(
        x,
        *[pivot[state].to_numpy() for state in STATE_ORDER],
        labels=STATE_ORDER,
        colors=[STATE_COLORS[state] for state in STATE_ORDER],
        linewidth=0,
    )
    ax_a.set(xlabel="Days since ICU admission", ylabel="State occupancy (%)", xlim=(0, 7), ylim=(0, 100))
    ax_a.set_title("Postoperative renal-state occupancy", loc="left", fontweight="bold")
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=5.3)
    panel_label(ax_a, "a")

    plot_cif_panel(
        ax_b,
        onset_cif,
        "First transition after AKI onset",
        [
            "Observed recovery",
            "Severe AKI onset/progression",
            "Live discharge",
            "In-hospital death",
        ],
        "Days since AKI onset",
        3,
    )
    ax_b.legend(fontsize=5.0, loc="upper left")
    panel_label(ax_b, "b")
    plot_cif_panel(
        ax_c,
        recurrence_cif,
        "Competing events after first recovery",
        ["Recurrent AKI", "Live discharge", "In-hospital death"],
        "Days since first recovery",
        5,
    )
    ax_c.legend(fontsize=5.0, loc="upper left")
    panel_label(ax_c, "c")
    save_figure(fig, "figure_v29_multistate_competing_risk")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True, sharex=True)
    comparisons = [
        ("CKD", ["No CKD", "CKD"]),
        ("Surgery type", ["Non-cardiac surgery", "Cardiac surgery"]),
    ]
    for row, (variable, groups) in enumerate(comparisons):
        for col, cause in enumerate(["Observed recovery", "Severe AKI onset/progression"]):
            ax = axes[row, col]
            for index, group_label in enumerate(groups):
                subset = onset_cif.loc[
                    onset_cif["group_variable"].eq(variable)
                    & onset_cif["group"].eq(group_label)
                    & onset_cif["cause"].eq(cause)
                ].sort_values("time_hours")
                color = ["#4C78A8", "#D97706"][index]
                ax.plot(subset["time_hours"] / 24, subset["cif_percent"], color=color, label=group_label)
                ax.fill_between(
                    subset["time_hours"] / 24,
                    subset["ci95_low_percent"],
                    subset["ci95_high_percent"],
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
            ax.set_title(f"{variable}: {cause}", loc="left", fontweight="bold")
            ax.set_ylabel("Cumulative incidence (%)")
            ax.set_xlabel("Days since AKI onset")
            ax.legend(fontsize=5.5)
            panel_label(ax, chr(ord("a") + row * 2 + col))
    save_figure(fig, "figure_v29_competing_risk_subgroups")


def write_contract_and_readme(
    cohort: pd.DataFrame,
    onset_data: pd.DataFrame,
    recurrence_data: pd.DataFrame,
    summary: pd.DataFrame,
    occupancy: pd.DataFrame,
) -> None:
    contract = """# v29 figure contract

- Core conclusion: postoperative kidney dysfunction follows a dynamic course in which recovery is common, severe progression accumulates early, recurrence remains observable after initial recovery, and hospital discharge/death materially compete with renal-state ascertainment.
- Evidence chain: panel a shows fixed-clock 0-7-day state occupancy; panel b shows cause-specific cumulative incidence after AKI onset; panel c shows recurrence and disposition after first observed recovery.
- Archetype: asymmetric quantitative composite with one hero occupancy panel.
- Backend: Python/matplotlib only.
- Export: 7.2-inch double-column layout; editable SVG/PDF text; 600-dpi TIFF and 300-dpi PNG; source data are the v29 CSV outputs.
- Statistical contract: Aalen-Johansen product-integral point estimates; patient-cluster bootstrap 95% percentile intervals; repeated admissions remain within subject clusters.
- Review risks: renal transitions are interval-observed at SCr measurement times; no unmeasured recovery is imputed; live discharge and in-hospital death are absorbing competing states; administrative follow-up ends 168 hours after ICU admission.
"""
    (OUT / "audit_v29_figure_contract.md").write_text(contract, encoding="utf-8")
    aki_n = int(bool_mask(cohort["aki_final"]).sum())
    recurrent_n = int(recurrence_data["first_event"].eq("Recurrent AKI").sum())
    max_difference = occupancy["aj_empirical_absolute_difference_percent"].max()
    readme = f"""# v29 multistate and competing-risk analysis

## Scope

This post-lock secondary analysis uses the {len(cohort):,}-stay strict evaluable MIMIC-IV cohort. It does not change the locked primary incident-AKI outcome or the deployed prediction models.

The multistate clock begins at ICU admission and ends at 168 hours. SCr-derived states change only at an observed measurement; no interpolation or unobserved recovery is imposed. Live hospital discharge and in-hospital death within 168 hours are absorbing states. The seven states are: No AKI, Stage 1 AKI, Severe AKI (stage 2/3), Recovered, Recurrent AKI, Live discharge, and In-hospital death.

Among {aki_n:,} incident-AKI stays, the first-transition competing-risk analysis starts at observed AKI onset. Causes are observed recovery, severe stage 2/3 progression, live discharge, and in-hospital death. A second competing-risk analysis starts at first observed recovery in {len(recurrence_data):,} stays and evaluates recurrent AKI, live discharge, and death.

Point estimates use the Aalen-Johansen product integral. Confidence intervals use {BOOTSTRAPS} subject-cluster bootstrap draws; all admissions belonging to a sampled patient are resampled together. The multistate product-integral estimates matched direct empirical state occupancy with a maximum absolute difference of {max_difference:.3g} percentage points.

## Interpretation boundary

These are descriptive observed-state estimates, not causal effects. Recovery and recurrence times are interval-observed at SCr measurements and therefore depend on measurement intensity. Discharge is treated as a competing absorbing state rather than as renal recovery. Follow-up beyond ICU day 7 is outside this analysis.

## Reproducibility

The transition-level data, occupancy source data, competing-risk datasets, subgroup estimates, figures, and independent validation report are written to this directory. The locked manuscript package is not modified automatically.
"""
    (OUT / "audit_v29_readme.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort, states = load_inputs()
    terminals = terminal_table(cohort)
    transitions = build_transition_process(cohort, states, terminals)
    assignments = empirical_assignments(cohort, transitions, GRID_HOURS)
    aj = aalen_johansen_occupancy(transitions, len(cohort), GRID_HOURS)
    empirical = (
        assignments.groupby(["time_hours", "state"]).size().rename("n").reset_index()
    )
    empirical["empirical_percent"] = 100 * empirical["n"] / len(cohort)
    occupancy = aj.merge(empirical, on=["time_hours", "state"], how="left", validate="one_to_one")
    occupancy["n"] = occupancy["n"].fillna(0).astype(int)
    occupancy["empirical_percent"] = occupancy["empirical_percent"].fillna(0)
    occupancy["aj_empirical_absolute_difference_percent"] = (
        occupancy["aj_percent"] - occupancy["empirical_percent"]
    ).abs()
    occupancy = occupancy.merge(
        bootstrap_occupancy(assignments, GRID_HOURS),
        on=["time_hours", "state"],
        validate="one_to_one",
    )

    onset_data, temporal_exclusions = competing_dataset_after_aki(
        cohort, transitions, terminals
    )
    recurrence_data = recurrence_dataset(cohort, transitions, terminals)
    onset_cif, recurrence_cif = all_cif_analyses(onset_data, recurrence_data)
    transitions_audit = transition_summary(transitions)
    summary = overall_summary(
        cohort, transitions, onset_data, recurrence_data, temporal_exclusions
    )

    transitions.to_csv(OUT / "cohort_v29_multistate_transitions.csv.gz", index=False, compression="gzip")
    assignments.to_csv(OUT / "cohort_v29_state_assignments.csv.gz", index=False, compression="gzip")
    onset_data.to_csv(OUT / "cohort_v29_competing_events_after_aki.csv", index=False)
    temporal_exclusions.to_csv(
        OUT / "audit_v29_postdisposition_aki_exclusions.csv", index=False
    )
    recurrence_data.to_csv(OUT / "cohort_v29_competing_events_after_recovery.csv", index=False)
    occupancy.to_csv(OUT / "analysis_v29_state_occupancy.csv", index=False)
    onset_cif.to_csv(OUT / "analysis_v29_cif_after_aki.csv", index=False)
    recurrence_cif.to_csv(OUT / "analysis_v29_cif_after_recovery.csv", index=False)
    transitions_audit.to_csv(OUT / "audit_v29_transition_counts.csv", index=False)
    summary.to_csv(OUT / "audit_v29_multistate_summary.csv", index=False)
    make_figures(occupancy, onset_cif, recurrence_cif)
    write_contract_and_readme(cohort, onset_data, recurrence_data, summary, occupancy)
    print(f"Wrote v29 multistate analysis to {OUT}")


if __name__ == "__main__":
    main()
