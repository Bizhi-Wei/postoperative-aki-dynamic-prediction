"""Characterize postoperative AKI severity, persistence, and early recovery.

This analysis deliberately goes beyond the binary ``aki_final`` outcome.  It
uses the serial serum-creatinine measurements that generated the v3 outcome to
derive:

* maximum KDIGO serum-creatinine stage and a stage 2/3 severe-AKI phenotype;
* rapid reversal versus persistent AKI (>48 h from onset, ADQI framework);
* recurrent AKI after an observed reversal;
* kidney-function status near the end of the seven-day/hospital observation
  window, with an explicit evaluability flag; and
* renal-replacement-therapy (RRT) use as a severity overlay/sensitivity outcome.

Absence of a creatinine measurement is never interpreted as recovery.  The
primary trajectory is based on serum creatinine so that it remains aligned with
the locked primary outcome; RRT is reported separately and in a sensitivity
severity stage.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MIMIC_ROOT = ROOT.parent
INPUT_COHORT = ROOT / "outputs" / "finalized_v3_1" / "cohort_v3_1_strict_main_evaluable.csv"
INPUT_SCR = ROOT / "outputs" / "cache_v3" / "creatinine_subject_labs.csv.gz"
PROCEDUREEVENTS = MIMIC_ROOT / "icu" / "procedureevents.csv"
OUT = ROOT / "outputs" / "modeling_v26_aki_severity_trajectories"
CACHE = ROOT / "outputs" / "cache_v26"

HORIZON = pd.Timedelta(days=7)
ROLLING_48H = pd.Timedelta(hours=48)
END_WINDOW = pd.Timedelta(hours=24)

RRT_ITEMIDS = {
    225436: "CRRT filter change",
    225441: "Hemodialysis",
    225802: "CRRT",
    225803: "CVVHD",
    225805: "Peritoneal dialysis",
    225809: "CVVHDF",
    225955: "SCUF",
}

STAGE_LABELS = {0: "No active AKI", 1: "Stage 1", 2: "Stage 2", 3: "Stage 3"}
STAGE_COLORS = {
    "No active AKI": "#D0D5DD",
    "Stage 1": "#7AA6C2",
    "Stage 2": "#E6A15A",
    "Stage 3": "#B84A4A",
    "Unobserved": "#F2F4F7",
}
TRAJECTORY_COLORS = {
    "Rapid sustained reversal": "#4C78A8",
    "Rapid reversal with recurrent AKI": "#8FB9D1",
    "Persistent AKI": "#D97706",
    "Indeterminate after 48 h": "#98A2B3",
    "Short observable follow-up": "#D0D5DD",
}


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
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(INPUT_COHORT, low_memory=False)
    required = {
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "dischtime",
        "baseline_scr_final",
        "aki_final",
        "aki_stage_final",
        "aki_onset_time_final",
    }
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise ValueError(f"Strict cohort is missing required columns: {missing}")
    if cohort["hadm_id"].duplicated().any() or cohort["stay_id"].duplicated().any():
        raise ValueError("The analysis cohort is not unique by hadm_id/stay_id")
    for column in ["intime", "outtime", "dischtime", "aki_onset_time_final"]:
        if column in cohort.columns:
            cohort[column] = pd.to_datetime(cohort[column], errors="coerce")
    cohort["aki_final"] = cohort["aki_final"].astype(str).str.lower().eq("true")
    cohort["aki_stage_final"] = pd.to_numeric(cohort["aki_stage_final"], errors="coerce")

    labs = pd.read_csv(INPUT_SCR, low_memory=False, parse_dates=["charttime"])
    labs["valuenum"] = pd.to_numeric(labs["valuenum"], errors="coerce")
    labs = labs.loc[
        labs["subject_id"].isin(cohort["subject_id"])
        & labs["charttime"].notna()
        & labs["valuenum"].gt(0)
        & labs["valuenum"].le(100)
    ].copy()
    return cohort, labs


def build_measurement_states(cohort: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    index = cohort[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "dischtime",
            "baseline_scr_final",
            "aki_final",
            "aki_onset_time_final",
        ]
    ].copy()
    labs = labs.rename(columns={"hadm_id": "lab_hadm_id"})
    joined = labs.merge(index, on="subject_id", how="inner", validate="many_to_many")
    compatible_admission = joined["lab_hadm_id"].isna() | joined["lab_hadm_id"].eq(joined["hadm_id"])
    joined = joined.loc[
        compatible_admission
        & joined["charttime"].ge(joined["intime"] - ROLLING_48H)
        & joined["charttime"].le(joined["intime"] + HORIZON)
    ].copy()
    joined = joined.sort_values(["stay_id", "charttime", "labevent_id"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for stay_id, group in joined.groupby("stay_id", sort=False):
        window: deque[tuple[pd.Timestamp, float]] = deque()
        baseline = float(group["baseline_scr_final"].iloc[0])
        intime = group["intime"].iloc[0]
        onset = group["aki_onset_time_final"].iloc[0]
        for lab in group.itertuples(index=False):
            while window and lab.charttime - window[0][0] > ROLLING_48H:
                window.popleft()
            absolute_48h = bool(window) and (
                float(lab.valuenum) - min(value for _, value in window) >= 0.3 - 1e-12
            )
            ratio = float(lab.valuenum) / baseline
            active_aki = bool(ratio >= 1.5 or absolute_48h)
            # An absolute SCr >=4 mg/dL upgrades *active AKI* to stage 3; it
            # must not label a stable high-baseline SCr as incident AKI.
            if active_aki and (ratio >= 3.0 or float(lab.valuenum) >= 4.0):
                stage = 3
            elif active_aki and ratio >= 2.0:
                stage = 2
            elif active_aki:
                stage = 1
            else:
                stage = 0
            if lab.charttime > intime:
                rows.append(
                    {
                        "subject_id": lab.subject_id,
                        "hadm_id": lab.hadm_id,
                        "stay_id": stay_id,
                        "charttime": lab.charttime,
                        "scr_mg_dl": float(lab.valuenum),
                        "hours_from_icu": (lab.charttime - intime).total_seconds() / 3600,
                        "hours_from_aki_onset": (
                            (lab.charttime - onset).total_seconds() / 3600
                            if pd.notna(onset)
                            else np.nan
                        ),
                        "scr_ratio_to_baseline": ratio,
                        "scr_delta_from_baseline": float(lab.valuenum) - baseline,
                        "aki_absolute_0_3_within_prior_48h": bool(absolute_48h),
                        "aki_active_by_kdigo_scr": bool(stage > 0),
                        "aki_stage_at_measurement": int(stage),
                        # Conservative recovery sensitivity: both baseline-referenced
                        # thresholds must have resolved.
                        "strict_baseline_aki_active": bool(
                            ratio >= 1.5 or float(lab.valuenum) - baseline >= 0.3 - 1e-12
                        ),
                    }
                )
            window.append((lab.charttime, float(lab.valuenum)))
    states = pd.DataFrame(rows)
    if states.empty:
        raise ValueError("No post-index creatinine states were generated")
    states["icu_day"] = np.floor(states["hours_from_icu"] / 24).astype(int) + 1
    return states


def extract_rrt_events(cohort: pd.DataFrame) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / "rrt_procedureevents_v26.csv.gz"
    stays = set(cohort["stay_id"].astype(int))
    if cache_path.exists():
        events = pd.read_csv(cache_path, low_memory=False, parse_dates=["starttime", "endtime"])
        if set(events["stay_id"].astype(int)).issubset(stays):
            return events

    retained: list[pd.DataFrame] = []
    usecols = ["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "statusdescription"]
    for chunk in pd.read_csv(PROCEDUREEVENTS, usecols=usecols, chunksize=500_000, low_memory=False):
        selected = chunk.loc[chunk["stay_id"].isin(stays) & chunk["itemid"].isin(RRT_ITEMIDS)].copy()
        if not selected.empty:
            retained.append(selected)
    events = pd.concat(retained, ignore_index=True) if retained else pd.DataFrame(columns=usecols)
    events["starttime"] = pd.to_datetime(events["starttime"], errors="coerce")
    events["endtime"] = pd.to_datetime(events["endtime"], errors="coerce")
    events["rrt_type"] = events["itemid"].map(RRT_ITEMIDS)
    events.to_csv(cache_path, index=False, compression="gzip")
    return events


def summarize_rrt(cohort: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=["stay_id", "rrt_within_7d", "rrt_first_time", "rrt_first_hours", "rrt_types"]
        )
    index = cohort[["stay_id", "intime"]]
    events = events.merge(index, on="stay_id", how="inner", validate="many_to_one")
    events["rrt_first_hours"] = (events["starttime"] - events["intime"]).dt.total_seconds() / 3600
    events = events.loc[events["rrt_first_hours"].gt(0) & events["rrt_first_hours"].le(168)].copy()
    if events.empty:
        return pd.DataFrame(
            columns=["stay_id", "rrt_within_7d", "rrt_first_time", "rrt_first_hours", "rrt_types"]
        )
    summary = (
        events.sort_values(["stay_id", "starttime"])
        .groupby("stay_id")
        .agg(
            rrt_first_time=("starttime", "first"),
            rrt_first_hours=("rrt_first_hours", "first"),
            rrt_types=("rrt_type", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
        )
        .reset_index()
    )
    summary["rrt_within_7d"] = True
    return summary


def derive_patient_phenotypes(
    cohort: pd.DataFrame, states: pd.DataFrame, rrt: pd.DataFrame
) -> pd.DataFrame:
    state_groups = {stay: group.sort_values(["charttime"]) for stay, group in states.groupby("stay_id")}
    records: list[dict[str, object]] = []
    for row in cohort.itertuples(index=False):
        group = state_groups.get(row.stay_id, states.iloc[0:0])
        record: dict[str, object] = {
            "stay_id": row.stay_id,
            "scr_measurements_7d": int(len(group)),
            "scr_observed_day1": bool(group["icu_day"].eq(1).any()) if len(group) else False,
            "scr_observed_day2": bool(group["icu_day"].eq(2).any()) if len(group) else False,
            "scr_observed_day3": bool(group["icu_day"].eq(3).any()) if len(group) else False,
            "scr_observed_days5_7": bool(group["icu_day"].between(5, 7).any()) if len(group) else False,
            "maximum_active_scr_stage_7d": int(group["aki_stage_at_measurement"].max())
            if len(group)
            else 0,
            "aki_duration_class": "No incident AKI",
            "persistence_evaluable": False,
            "persistent_aki_scr": pd.NA,
            "rapid_reversal_within_48h": pd.NA,
            "recurrent_aki_after_first_reversal": pd.NA,
            "first_recovery_time": pd.NaT,
            "first_recovery_hours_from_onset": np.nan,
            "first_strict_baseline_recovery_time": pd.NaT,
            "first_strict_baseline_recovery_hours": np.nan,
            "end_observation_time": min(row.intime + HORIZON, row.dischtime)
            if pd.notna(row.dischtime)
            else row.intime + HORIZON,
            "end_recovery_evaluable": False,
            "end_scr_time": pd.NaT,
            "end_scr_mg_dl": np.nan,
            "end_scr_ratio_to_baseline": np.nan,
            "end_aki_stage_observed": pd.NA,
            "recovered_at_end_by_kdigo_scr": pd.NA,
            "recovered_at_end_strict_baseline": pd.NA,
            "partial_recovery_at_end": pd.NA,
            "renal_trajectory_group": "No incident AKI",
        }
        if not bool(row.aki_final):
            records.append(record)
            continue

        onset = row.aki_onset_time_final
        episode = group.loc[group["charttime"].ge(onset)].copy()
        recovered = episode.loc[~episode["aki_active_by_kdigo_scr"]]
        first_recovery = recovered["charttime"].min() if not recovered.empty else pd.NaT
        first_recovery_hours = (
            (first_recovery - onset).total_seconds() / 3600 if pd.notna(first_recovery) else np.nan
        )
        strict_recovered = episode.loc[~episode["strict_baseline_aki_active"]]
        first_strict = strict_recovered["charttime"].min() if not strict_recovered.empty else pd.NaT
        first_strict_hours = (
            (first_strict - onset).total_seconds() / 3600 if pd.notna(first_strict) else np.nan
        )
        rapid = bool(pd.notna(first_recovery) and first_recovery_hours <= 48 + 1e-12)
        recurrent = bool(
            pd.notna(first_recovery)
            and episode.loc[episode["charttime"].gt(first_recovery), "aki_active_by_kdigo_scr"].any()
        )
        active_after_48 = episode.loc[
            episode["hours_from_aki_onset"].ge(48) & episode["aki_active_by_kdigo_scr"]
        ]
        observed_after_48 = bool(episode["hours_from_aki_onset"].ge(48).any())
        if rapid:
            duration_class = (
                "Rapid reversal with recurrent AKI" if recurrent else "Rapid sustained reversal"
            )
            persistence_evaluable = True
            persistent = False
        elif not active_after_48.empty:
            duration_class = "Persistent AKI"
            persistence_evaluable = True
            persistent = True
        elif observed_after_48:
            duration_class = "Indeterminate after 48 h"
            persistence_evaluable = False
            persistent = pd.NA
        else:
            duration_class = "Short observable follow-up"
            persistence_evaluable = False
            persistent = pd.NA

        end_time = record["end_observation_time"]
        end_window = episode.loc[
            episode["charttime"].le(end_time)
            & episode["charttime"].ge(end_time - END_WINDOW)
        ]
        if not end_window.empty:
            end_row = end_window.sort_values("charttime").iloc[-1]
            end_evaluable = True
            end_stage = int(end_row["aki_stage_at_measurement"])
            end_recovered = bool(end_stage == 0)
            end_strict = bool(not end_row["strict_baseline_aki_active"])
            partial = bool(0 < end_stage < int(record["maximum_active_scr_stage_7d"]))
        else:
            end_row = None
            end_evaluable = False
            end_stage = pd.NA
            end_recovered = pd.NA
            end_strict = pd.NA
            partial = pd.NA

        if duration_class == "Rapid sustained reversal":
            trajectory = (
                "Rapid sustained reversal"
                if end_evaluable and end_recovered
                else "Rapid reversal; end status unconfirmed"
            )
        elif duration_class == "Rapid reversal with recurrent AKI":
            trajectory = "Rapid reversal with recurrent AKI"
        elif duration_class == "Persistent AKI":
            if not end_evaluable:
                trajectory = "Persistent AKI; recovery indeterminate"
            elif end_recovered:
                trajectory = "Persistent AKI with recovery"
            else:
                trajectory = "Persistent AKI without recovery"
        elif end_evaluable and end_recovered:
            trajectory = "Duration indeterminate; recovered at end"
        elif end_evaluable:
            trajectory = "Duration indeterminate; not recovered at end"
        else:
            trajectory = "Duration and recovery indeterminate"

        record.update(
            {
                "aki_duration_class": duration_class,
                "persistence_evaluable": persistence_evaluable,
                "persistent_aki_scr": persistent,
                "rapid_reversal_within_48h": rapid,
                "recurrent_aki_after_first_reversal": recurrent,
                "first_recovery_time": first_recovery,
                "first_recovery_hours_from_onset": first_recovery_hours,
                "first_strict_baseline_recovery_time": first_strict,
                "first_strict_baseline_recovery_hours": first_strict_hours,
                "end_recovery_evaluable": end_evaluable,
                "end_scr_time": end_row["charttime"] if end_row is not None else pd.NaT,
                "end_scr_mg_dl": end_row["scr_mg_dl"] if end_row is not None else np.nan,
                "end_scr_ratio_to_baseline": (
                    end_row["scr_ratio_to_baseline"] if end_row is not None else np.nan
                ),
                "end_aki_stage_observed": end_stage,
                "recovered_at_end_by_kdigo_scr": end_recovered,
                "recovered_at_end_strict_baseline": end_strict,
                "partial_recovery_at_end": partial,
                "renal_trajectory_group": trajectory,
            }
        )
        records.append(record)

    phenotypes = cohort.merge(pd.DataFrame(records), on="stay_id", how="left", validate="one_to_one")
    phenotypes = phenotypes.merge(rrt, on="stay_id", how="left", validate="one_to_one")
    # De-fragment the wide inherited cohort before adding final phenotype fields.
    phenotypes = phenotypes.copy()
    phenotypes["rrt_within_7d"] = phenotypes["rrt_within_7d"].fillna(False).astype(bool)
    phenotypes["severe_aki_scr_stage2_3"] = phenotypes["maximum_active_scr_stage_7d"].ge(2)
    phenotypes["severity_stage_discordant_vs_locked_v3"] = (
        phenotypes["maximum_active_scr_stage_7d"].astype(int)
        != phenotypes["aki_stage_final"].fillna(0).astype(int)
    )
    phenotypes["aki_stage_scr_or_rrt_7d"] = np.maximum(
        phenotypes["maximum_active_scr_stage_7d"].fillna(0).astype(int),
        np.where(phenotypes["rrt_within_7d"], 3, 0),
    )
    phenotypes["severe_aki_scr_or_rrt"] = phenotypes["aki_stage_scr_or_rrt_7d"].ge(2)
    phenotypes["severity_group"] = phenotypes["maximum_active_scr_stage_7d"].map(
        {0: "No incident AKI", 1: "Stage 1 AKI", 2: "Stage 2 AKI", 3: "Stage 3 AKI"}
    )
    return phenotypes


def count_table(series: pd.Series, denominator: int, dimension: str) -> pd.DataFrame:
    table = series.value_counts(dropna=False).rename_axis("category").reset_index(name="n")
    table["dimension"] = dimension
    table["denominator_n"] = denominator
    table["percent"] = 100 * table["n"] / denominator
    return table[["dimension", "category", "n", "denominator_n", "percent"]]


def q(series: pd.Series, probability: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(probability)) if len(values) else np.nan


def outcome_summary(data: pd.DataFrame, group_column: str, dimension: str) -> pd.DataFrame:
    rows = []
    for group, subset in data.groupby(group_column, dropna=False):
        row: dict[str, object] = {"dimension": dimension, "group": group, "n": len(subset)}
        for outcome in ["icu_death", "hosp_death", "death_90d", "death_365d"]:
            if outcome in subset.columns:
                values = pd.to_numeric(subset[outcome], errors="coerce")
                row[f"{outcome}_n"] = int(values.sum())
                row[f"{outcome}_percent"] = 100 * float(values.mean())
        for variable, prefix in [("los", "icu_los_days"), ("hosp_los", "hospital_los_days")]:
            if variable in subset.columns:
                row[f"{prefix}_median"] = q(subset[variable], 0.5)
                row[f"{prefix}_q1"] = q(subset[variable], 0.25)
                row[f"{prefix}_q3"] = q(subset[variable], 0.75)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_crude_contrasts(phenotypes: pd.DataFrame, draws: int = 1000) -> pd.DataFrame:
    """Subject-cluster bootstrap for prespecified descriptive phenotype contrasts."""
    aki = phenotypes.loc[phenotypes["aki_final"]].copy()
    contrasts = [
        (
            "Severe stage 2/3 vs stage 1",
            aki["maximum_active_scr_stage_7d"].ge(2),
            aki["maximum_active_scr_stage_7d"].eq(1),
        ),
        (
            "Persistent AKI vs rapid sustained reversal",
            aki["aki_duration_class"].eq("Persistent AKI"),
            aki["aki_duration_class"].eq("Rapid sustained reversal"),
        ),
        (
            "Persistent AKI without recovery vs rapid sustained reversal",
            aki["renal_trajectory_group"].eq("Persistent AKI without recovery"),
            aki["renal_trajectory_group"].eq("Rapid sustained reversal"),
        ),
        (
            "Rapid reversal with recurrence vs rapid sustained reversal",
            aki["renal_trajectory_group"].eq("Rapid reversal with recurrent AKI"),
            aki["renal_trajectory_group"].eq("Rapid sustained reversal"),
        ),
    ]
    outcomes = [
        ("hosp_death", "binary"),
        ("death_90d", "binary"),
        ("los", "continuous"),
        ("hosp_los", "continuous"),
    ]
    rows: list[dict[str, object]] = []
    for contrast_index, (label, exposed_mask, reference_mask) in enumerate(contrasts):
        subset = aki.loc[exposed_mask | reference_mask].copy()
        subset["exposed"] = exposed_mask.loc[subset.index].astype(bool).to_numpy()
        local = subset.reset_index(drop=True)
        # Re-create local row indices after reset.
        subject_rows = [group.index.to_numpy() for _, group in local.groupby("subject_id")]
        rng = np.random.default_rng(2600 + contrast_index)
        boot: dict[tuple[str, str], list[float]] = {}
        for outcome, outcome_type in outcomes:
            boot[(outcome, "primary")] = []
            if outcome_type == "binary":
                boot[(outcome, "secondary")] = []
        for _ in range(draws):
            sampled_subjects = rng.integers(0, len(subject_rows), size=len(subject_rows))
            idx = np.concatenate([subject_rows[i] for i in sampled_subjects])
            sampled = local.iloc[idx]
            exp = sampled.loc[sampled["exposed"]]
            ref = sampled.loc[~sampled["exposed"]]
            if exp.empty or ref.empty:
                continue
            for outcome, outcome_type in outcomes:
                exp_values = pd.to_numeric(exp[outcome], errors="coerce").dropna()
                ref_values = pd.to_numeric(ref[outcome], errors="coerce").dropna()
                if exp_values.empty or ref_values.empty:
                    continue
                if outcome_type == "binary":
                    exp_rate, ref_rate = float(exp_values.mean()), float(ref_values.mean())
                    boot[(outcome, "primary")].append(100 * (exp_rate - ref_rate))
                    boot[(outcome, "secondary")].append(exp_rate / ref_rate if ref_rate > 0 else np.nan)
                else:
                    boot[(outcome, "primary")].append(float(exp_values.median() - ref_values.median()))

        exposed = local.loc[local["exposed"]]
        reference = local.loc[~local["exposed"]]
        for outcome, outcome_type in outcomes:
            exp_values = pd.to_numeric(exposed[outcome], errors="coerce").dropna()
            ref_values = pd.to_numeric(reference[outcome], errors="coerce").dropna()
            if outcome_type == "binary":
                exp_rate, ref_rate = float(exp_values.mean()), float(ref_values.mean())
                measures = [
                    ("risk difference, percentage points", 100 * (exp_rate - ref_rate), "primary"),
                    ("risk ratio", exp_rate / ref_rate if ref_rate > 0 else np.nan, "secondary"),
                ]
                exposed_value, reference_value = 100 * exp_rate, 100 * ref_rate
            else:
                measures = [
                    ("median difference, days", float(exp_values.median() - ref_values.median()), "primary")
                ]
                exposed_value, reference_value = float(exp_values.median()), float(ref_values.median())
            for measure, estimate, key in measures:
                values = np.asarray(boot[(outcome, key)], dtype=float)
                values = values[np.isfinite(values)]
                rows.append(
                    {
                        "contrast": label,
                        "exposed_n": len(exposed),
                        "reference_n": len(reference),
                        "outcome": outcome,
                        "exposed_value": exposed_value,
                        "reference_value": reference_value,
                        "effect_measure": measure,
                        "estimate": estimate,
                        "ci95_low": float(np.quantile(values, 0.025)),
                        "ci95_high": float(np.quantile(values, 0.975)),
                        "bootstrap_successful_n": len(values),
                        "inference_scope": "crude descriptive association; subject-cluster bootstrap",
                    }
                )
    return pd.DataFrame(rows)


def make_audits(phenotypes: pd.DataFrame, states: pd.DataFrame) -> dict[str, pd.DataFrame]:
    aki = phenotypes.loc[phenotypes["aki_final"]].copy()
    severity = pd.concat(
        [
            count_table(
                phenotypes["severity_group"],
                len(phenotypes),
                "maximum active-episode SCr stage",
            ),
            count_table(
                phenotypes["severe_aki_scr_stage2_3"].map({True: "Severe AKI (stage 2/3)", False: "No severe AKI"}),
                len(phenotypes),
                "severe SCr AKI",
            ),
            count_table(
                phenotypes["rrt_within_7d"].map({True: "RRT within 7 d", False: "No RRT within 7 d"}),
                len(phenotypes),
                "RRT severity overlay",
            ),
            count_table(
                phenotypes["aki_stage_scr_or_rrt_7d"].map(
                    {0: "No AKI/RRT", 1: "Stage 1", 2: "Stage 2", 3: "Stage 3 including RRT"}
                ),
                len(phenotypes),
                "SCr-or-RRT sensitivity stage",
            ),
            count_table(
                phenotypes["severe_aki_scr_or_rrt"].map(
                    {True: "Severe AKI (SCr stage 2/3 or RRT)", False: "No severe AKI"}
                ),
                len(phenotypes),
                "severe SCr-or-RRT sensitivity",
            ),
        ],
        ignore_index=True,
    )

    trajectories = pd.concat(
        [
            count_table(aki["aki_duration_class"], len(aki), "48 h duration phenotype among AKI"),
            count_table(aki["renal_trajectory_group"], len(aki), "renal trajectory among AKI"),
            count_table(
                aki.loc[aki["end_recovery_evaluable"], "end_aki_stage_observed"].map(STAGE_LABELS),
                int(aki["end_recovery_evaluable"].sum()),
                "end status among evaluable AKI",
            ),
            count_table(
                aki.loc[aki["end_recovery_evaluable"], "recovered_at_end_strict_baseline"].map(
                    {True: "Recovered (strict baseline definition)", False: "Not recovered"}
                ),
                int(aki["end_recovery_evaluable"].sum()),
                "strict recovery sensitivity among evaluable AKI",
            ),
        ],
        ignore_index=True,
    )

    obs_rows: list[dict[str, object]] = []
    groups = {
        "Full cohort": phenotypes,
        "Any incident AKI": aki,
        "Stage 1 AKI": aki.loc[aki["maximum_active_scr_stage_7d"].eq(1)],
        "Stage 2/3 AKI": aki.loc[aki["maximum_active_scr_stage_7d"].ge(2)],
    }
    for label, subset in groups.items():
        obs_rows.append(
            {
                "group": label,
                "n": len(subset),
                "scr_measurements_7d_median": q(subset["scr_measurements_7d"], 0.5),
                "scr_measurements_7d_q1": q(subset["scr_measurements_7d"], 0.25),
                "scr_measurements_7d_q3": q(subset["scr_measurements_7d"], 0.75),
                "scr_observed_day1_percent": 100 * subset["scr_observed_day1"].mean(),
                "scr_observed_day2_percent": 100 * subset["scr_observed_day2"].mean(),
                "scr_observed_day3_percent": 100 * subset["scr_observed_day3"].mean(),
                "scr_observed_days5_7_percent": 100 * subset["scr_observed_days5_7"].mean(),
                "persistence_evaluable_percent": (
                    100 * subset["persistence_evaluable"].mean() if label != "Full cohort" else np.nan
                ),
                "end_recovery_evaluable_percent": (
                    100 * subset["end_recovery_evaluable"].mean() if label != "Full cohort" else np.nan
                ),
            }
        )
    observability = pd.DataFrame(obs_rows)

    daily_rows: list[dict[str, object]] = []
    for population, ids in [("Full cohort", phenotypes["stay_id"]), ("Incident AKI", aki["stay_id"])]:
        denominator = len(ids)
        subset = states.loc[states["stay_id"].isin(ids) & states["icu_day"].between(1, 7)]
        last = (
            subset.sort_values(["stay_id", "icu_day", "charttime"])
            .groupby(["stay_id", "icu_day"])
            .tail(1)
        )
        for day in range(1, 8):
            day_values = last.loc[last["icu_day"].eq(day), "aki_stage_at_measurement"].map(STAGE_LABELS)
            counts = day_values.value_counts()
            observed = int(counts.sum())
            for category in ["No active AKI", "Stage 1", "Stage 2", "Stage 3"]:
                n = int(counts.get(category, 0))
                daily_rows.append(
                    {
                        "population": population,
                        "icu_day": day,
                        "state": category,
                        "n": n,
                        "denominator_n": denominator,
                        "percent_of_population": 100 * n / denominator,
                        "percent_of_observed": 100 * n / observed if observed else np.nan,
                    }
                )
            daily_rows.append(
                {
                    "population": population,
                    "icu_day": day,
                    "state": "Unobserved",
                    "n": denominator - observed,
                    "denominator_n": denominator,
                    "percent_of_population": 100 * (denominator - observed) / denominator,
                    "percent_of_observed": np.nan,
                }
            )
    daily = pd.DataFrame(daily_rows)

    aki_states = states.loc[states["stay_id"].isin(aki["stay_id"]) & states["icu_day"].between(1, 7)]
    daily_last = (
        aki_states.sort_values(["stay_id", "icu_day", "charttime"])
        .groupby(["stay_id", "icu_day"])
        .tail(1)[["stay_id", "icu_day", "aki_stage_at_measurement"]]
    )
    transition_rows = []
    for day in range(1, 7):
        left = daily_last.loc[daily_last["icu_day"].eq(day), ["stay_id", "aki_stage_at_measurement"]]
        right = daily_last.loc[daily_last["icu_day"].eq(day + 1), ["stay_id", "aki_stage_at_measurement"]]
        pair = left.merge(right, on="stay_id", suffixes=("_from", "_to"))
        counts = pair.groupby(["aki_stage_at_measurement_from", "aki_stage_at_measurement_to"]).size()
        for from_stage in range(4):
            denom = int((pair["aki_stage_at_measurement_from"] == from_stage).sum())
            for to_stage in range(4):
                n = int(counts.get((from_stage, to_stage), 0))
                transition_rows.append(
                    {
                        "from_icu_day": day,
                        "to_icu_day": day + 1,
                        "from_state": STAGE_LABELS[from_stage],
                        "to_state": STAGE_LABELS[to_stage],
                        "n": n,
                        "from_state_denominator_n": denom,
                        "row_percent": 100 * n / denom if denom else np.nan,
                    }
                )
    transitions = pd.DataFrame(transition_rows)

    outcomes = pd.concat(
        [
            outcome_summary(
                phenotypes,
                "severity_group",
                "maximum active-episode SCr severity",
            ),
            outcome_summary(aki, "aki_duration_class", "48 h duration phenotype"),
            outcome_summary(aki, "renal_trajectory_group", "renal trajectory"),
        ],
        ignore_index=True,
    )
    return {
        "severity": severity,
        "trajectories": trajectories,
        "observability": observability,
        "daily": daily,
        "transitions": transitions,
        "outcomes": outcomes,
    }


def make_main_figure(phenotypes: pd.DataFrame, audits: dict[str, pd.DataFrame]) -> None:
    aki = phenotypes.loc[phenotypes["aki_final"]].copy()
    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.18])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # a: full-cohort severity distribution
    severity_order = ["No incident AKI", "Stage 1 AKI", "Stage 2 AKI", "Stage 3 AKI"]
    severity_colors = ["#D0D5DD", "#7AA6C2", "#E6A15A", "#B84A4A"]
    counts = phenotypes["severity_group"].value_counts()
    left = 0.0
    for label, color in zip(severity_order, severity_colors):
        value = 100 * counts.get(label, 0) / len(phenotypes)
        ax_a.barh([0], [value], left=left, color=color, height=0.42, label=label)
        if value >= 4:
            ax_a.text(left + value / 2, 0, f"{value:.1f}%", ha="center", va="center", fontsize=6.2)
        left += value
    ax_a.set_xlim(0, 100)
    ax_a.set_yticks([])
    ax_a.set_xlabel("Percentage of strict cohort")
    ax_a.set_title(f"Maximum active-episode SCr severity (n={len(phenotypes):,})", loc="left", fontweight="bold")
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2, frameon=False)
    panel_label(ax_a, "a")

    # b: duration phenotype among AKI
    duration_order = list(TRAJECTORY_COLORS)
    counts = aki["aki_duration_class"].value_counts()
    left = 0.0
    for label in duration_order:
        value = 100 * counts.get(label, 0) / len(aki)
        ax_b.barh([0], [value], left=left, color=TRAJECTORY_COLORS[label], height=0.42, label=label)
        if value >= 5:
            ax_b.text(left + value / 2, 0, f"{value:.1f}%", ha="center", va="center", fontsize=6.2)
        left += value
    ax_b.set_xlim(0, 100)
    ax_b.set_yticks([])
    ax_b.set_xlabel("Percentage of incident AKI")
    ax_b.set_title(f"AKI duration phenotype (n={len(aki):,})", loc="left", fontweight="bold")
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=1, frameon=False)
    panel_label(ax_b, "b")

    # c: end recovery status by maximum severity
    severity = ["Stage 1 AKI", "Stage 2 AKI", "Stage 3 AKI"]
    y = np.arange(len(severity))
    recovered, partial, not_recovered, coverage = [], [], [], []
    for label in severity:
        subset = aki.loc[aki["severity_group"].eq(label)]
        evaluable = subset.loc[subset["end_recovery_evaluable"]]
        coverage.append(100 * len(evaluable) / len(subset))
        recovered.append(100 * evaluable["end_aki_stage_observed"].eq(0).mean())
        partial.append(100 * evaluable["partial_recovery_at_end"].eq(True).mean())
        not_recovered.append(
            100
            * (
                ~evaluable["end_aki_stage_observed"].eq(0)
                & ~evaluable["partial_recovery_at_end"].eq(True)
            ).mean()
        )
    ax_c.barh(y, recovered, color="#4C78A8", label="Complete reversal")
    ax_c.barh(y, partial, left=recovered, color="#F2C879", label="Partial stage improvement")
    ax_c.barh(
        y,
        not_recovered,
        left=np.asarray(recovered) + np.asarray(partial),
        color="#B84A4A",
        label="No stage improvement",
    )
    ax_c.set_yticks(y, severity)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 100)
    ax_c.set_xlabel("Percentage among end-status evaluable patients")
    ax_c.set_title("Observed end-of-window kidney status", loc="left", fontweight="bold")
    for i, cov in enumerate(coverage):
        ax_c.text(101, i, f"coverage {cov:.1f}%", va="center", fontsize=6.2, clip_on=False)
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    panel_label(ax_c, "c")

    # d: descriptive hospital mortality by clinically interpretable trajectory
    display_groups = [
        "Rapid sustained reversal",
        "Rapid reversal with recurrent AKI",
        "Persistent AKI with recovery",
        "Persistent AKI without recovery",
    ]
    labels = ["Rapid sustained", "Rapid + recurrent", "Persistent + recovery", "Persistent + no recovery"]
    values, ns = [], []
    for group in display_groups:
        subset = aki.loc[aki["renal_trajectory_group"].eq(group)]
        values.append(100 * subset["hosp_death"].mean() if len(subset) else np.nan)
        ns.append(len(subset))
    bars = ax_d.barh(np.arange(len(labels)), values, color=["#4C78A8", "#8FB9D1", "#D97706", "#B84A4A"])
    ax_d.set_yticks(np.arange(len(labels)), labels)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("In-hospital mortality (%)")
    ax_d.set_title("Clinical outcome by renal trajectory", loc="left", fontweight="bold")
    for bar, value, n in zip(bars, values, ns):
        ax_d.text(value + 0.25, bar.get_y() + bar.get_height() / 2, f"{value:.1f}% (n={n:,})", va="center", fontsize=6.2)
    ax_d.set_xlim(0, max(values) * 1.33 if values else 10)
    panel_label(ax_d, "d")
    save_figure(fig, "figure_v26_severity_recovery_phenotypes")


def make_daily_figure(audits: dict[str, pd.DataFrame]) -> None:
    daily = audits["daily"].loc[audits["daily"]["population"].eq("Incident AKI")]
    transitions = audits["transitions"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [1.28, 0.92]}, constrained_layout=True)
    ax_a, ax_b = axes
    bottom = np.zeros(7)
    for state in ["No active AKI", "Stage 1", "Stage 2", "Stage 3", "Unobserved"]:
        subset = daily.loc[daily["state"].eq(state)].sort_values("icu_day")
        values = subset["percent_of_population"].to_numpy()
        ax_a.bar(subset["icu_day"], values, bottom=bottom, color=STAGE_COLORS[state], width=0.78, label=state)
        bottom += values
    ax_a.set_ylim(0, 100)
    ax_a.set_xticks(range(1, 8))
    ax_a.set_xlabel("ICU day")
    ax_a.set_ylabel("Percentage of incident-AKI patients")
    ax_a.set_title("Observed daily kidney state and measurement coverage", loc="left", fontweight="bold")
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    panel_label(ax_a, "a")

    pooled = transitions.groupby(["from_state", "to_state"], as_index=False)["n"].sum()
    matrix = pooled.pivot(index="from_state", columns="to_state", values="n").reindex(
        index=[STAGE_LABELS[i] for i in range(4)], columns=[STAGE_LABELS[i] for i in range(4)]
    ).fillna(0)
    row_percent = matrix.div(matrix.sum(axis=1), axis=0) * 100
    im = ax_b.imshow(row_percent.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="equal")
    ax_b.set_xticks(range(4), ["No AKI", "S1", "S2", "S3"])
    ax_b.set_yticks(range(4), ["No AKI", "S1", "S2", "S3"])
    ax_b.set_xlabel("Next observed ICU day")
    ax_b.set_ylabel("Current observed ICU day")
    ax_b.set_title("Transitions between consecutive observed days", loc="left", fontweight="bold")
    for i in range(4):
        for j in range(4):
            value = row_percent.iloc[i, j]
            ax_b.text(j, i, f"{value:.1f}%", ha="center", va="center", fontsize=6, color="white" if value > 55 else "#101828")
    fig.colorbar(im, ax=ax_b, fraction=0.045, pad=0.04, label="Row percentage")
    panel_label(ax_b, "b")
    save_figure(fig, "figure_v26_daily_scr_state_trajectories")


def format_n_pct(n: int, denominator: int) -> str:
    return f"{n:,} ({100*n/denominator:.1f}%)"


def write_reports(phenotypes: pd.DataFrame, audits: dict[str, pd.DataFrame]) -> None:
    aki = phenotypes.loc[phenotypes["aki_final"]]
    severe = int(phenotypes["severe_aki_scr_stage2_3"].sum())
    duration = aki["aki_duration_class"].value_counts()
    persistent_n = int(duration.get("Persistent AKI", 0))
    rapid_sustained_n = int(duration.get("Rapid sustained reversal", 0))
    rapid_recurrent_n = int(duration.get("Rapid reversal with recurrent AKI", 0))
    end_eval = aki.loc[aki["end_recovery_evaluable"]]
    recovered_end = int(end_eval["recovered_at_end_by_kdigo_scr"].sum())
    strict_recovered_end = int(end_eval["recovered_at_end_strict_baseline"].sum())
    rrt_n = int(phenotypes["rrt_within_7d"].sum())
    severe_scr_rrt_n = int(phenotypes["severe_aki_scr_or_rrt"].sum())
    severity_discordant_n = int(phenotypes["severity_stage_discordant_vs_locked_v3"].sum())
    persistent_eval_n = int(aki["persistence_evaluable"].sum())
    persistent_nonrec = int((aki["renal_trajectory_group"] == "Persistent AKI without recovery").sum())
    persistent_rec = int((aki["renal_trajectory_group"] == "Persistent AKI with recovery").sum())

    brief = f"""# v26 AKI severity and recovery-trajectory analysis

## Key results

- The strict evaluable cohort contained **{len(phenotypes):,}** admissions and **{len(aki):,}** incident SCr-AKI events.
- Severe SCr AKI (KDIGO stage 2 or 3) occurred in **{format_n_pct(severe, len(phenotypes))}** of the full cohort and **{format_n_pct(severe, len(aki))}** of AKI cases.
- Active-episode restaging differed from the locked peak-based v3 stage in **{severity_discordant_n:,}** rows; the legacy field is retained for audit, while severity analyses use `maximum_active_scr_stage_7d`.
- Seven-day ICU RRT procedure evidence was present in **{format_n_pct(rrt_n, len(phenotypes))}**. Adding RRT as KDIGO stage 3 increased the severe-outcome sensitivity count to **{format_n_pct(severe_scr_rrt_n, len(phenotypes))}**; this overlay does not replace the locked SCr primary outcome.
- AKI persistence was classifiable in **{format_n_pct(persistent_eval_n, len(aki))}**. Among all AKI cases, rapid sustained reversal occurred in **{format_n_pct(rapid_sustained_n, len(aki))}**, rapid reversal followed by recurrent AKI in **{format_n_pct(rapid_recurrent_n, len(aki))}**, and persistent AKI in **{format_n_pct(persistent_n, len(aki))}**.
- End-of-observation recovery was evaluable in **{format_n_pct(len(end_eval), len(aki))}** using a creatinine measured within 24 h of hospital discharge or ICU day 7, whichever came first. Complete KDIGO-SCr reversal was observed in **{format_n_pct(recovered_end, len(end_eval))}**; the conservative baseline-referenced sensitivity definition yielded **{format_n_pct(strict_recovered_end, len(end_eval))}**.
- Among all AKI cases, **{persistent_rec:,}** had persistent AKI with observed recovery and **{persistent_nonrec:,}** had persistent AKI without recovery at the evaluable end point.

## Interpretation

The binary any-AKI outcome combines clinically distinct phenotypes. Stage 2/3 AKI was uncommon relative to stage 1 but was much more often persistent. Persistent AKI without observed recovery had the highest crude mortality and longest stays. These comparisons are descriptive associations, not causal effects; differential creatinine measurement remains an explicit source of selection.

## Recommended manuscript role

Use this as a prespecified secondary phenotype analysis after the locked any-AKI model results. The strongest next modeling targets are (1) stage 2/3 SCr AKI in the full risk set and (2) persistent/nonrecovered AKI among incident-AKI patients. The current output first establishes event counts and outcome observability before any additional model is trained.
"""
    (OUT / "audit_v26_results_brief.md").write_text(brief, encoding="utf-8")

    readme = """# v26 AKI severity and recovery trajectories

## Definitions

- **Severe AKI:** maximum *active-episode* KDIGO SCr stage 2 or 3 within seven days after ICU admission. An SCr >=4 mg/dL upgrades active AKI to stage 3 but does not create incident AKI in a stable high-baseline patient. The legacy `aki_stage_final` is retained for audit; secondary severity analyses use `maximum_active_scr_stage_7d`. RRT within seven days is reported separately and upgrades severity to stage 3 only in the sensitivity field `aki_stage_scr_or_rrt_7d`.
- **Active SCr AKI at a measurement:** SCr ratio >=1.5 versus baseline, SCr >=4.0 mg/dL for stage 3, or a >=0.3 mg/dL rise from a prior SCr in the preceding 48 h.
- **Rapid reversal:** first observed measurement without active KDIGO SCr AKI within 48 h after AKI onset.
- **Persistent AKI:** no rapid reversal and an observed AKI-positive SCr measurement at least 48 h after onset.
- **Recurrent AKI:** an observed AKI-positive SCr after the first observed reversal. This is not assumed to be a distinct episode unless reversal is sustained; the flag is descriptive.
- **End recovery:** the last SCr within 24 h before the earlier of hospital discharge or ICU day 7 shows no active KDIGO SCr AKI.
- **Strict recovery sensitivity:** the same end SCr is both <1.5 times baseline and <0.3 mg/dL above baseline.

## Observability safeguards

No SCr measurement is never coded as recovery. Persistence and end recovery have separate evaluability flags. Patients with no suitable end-window SCr remain unclassified. Daily state plots include an explicit unobserved segment, and transitions use only consecutive ICU days with a measured SCr on both days.

## Files

- `cohort_v26_strict_aki_severity_recovery.csv`: patient-level phenotype file.
- `creatinine_measurement_states_v26.csv.gz`: measurement-level state file.
- `audit_v26_severity_summary.csv`: severity and RRT distributions.
- `audit_v26_recovery_trajectory_summary.csv`: duration and recovery distributions.
- `audit_v26_observability_summary.csv`: SCr coverage and phenotype evaluability.
- `audit_v26_clinical_outcomes_by_phenotype.csv`: crude mortality and length-of-stay summaries.
- `audit_v26_crude_outcome_contrasts_bootstrap.csv`: crude phenotype contrasts with subject-cluster bootstrap 95% CIs.
- `audit_v26_daily_state_distribution.csv`: daily measured-state source data.
- `audit_v26_observed_transition_matrix.csv`: transitions between consecutive observed days.
- `figure_v26_*`: publication figures in PNG, PDF, SVG and TIFF.

## Scope

This is a secondary phenotyping and descriptive outcome analysis. It does not alter the locked primary AKI labels, retrain the deployed models, or make causal claims about recovery.
"""
    (OUT / "audit_v26_readme.md").write_text(readme, encoding="utf-8")


def validate_outputs(phenotypes: pd.DataFrame, states: pd.DataFrame) -> list[str]:
    checks: list[str] = []
    if phenotypes["stay_id"].duplicated().any():
        raise AssertionError("Duplicate stay_id in phenotype output")
    checks.append("PASS: one row per stay_id in the phenotype output")
    original_stage = phenotypes["aki_stage_final"].value_counts().sort_index()
    if int(original_stage.sum()) != len(phenotypes):
        raise AssertionError("Missing original stage labels")
    checks.append("PASS: locked v3 SCr AKI stage labels were retained for every row")
    if phenotypes.loc[~phenotypes["aki_final"], "persistent_aki_scr"].notna().any():
        raise AssertionError("Persistence assigned to a non-AKI row")
    checks.append("PASS: persistence/reversal phenotypes are restricted to incident-AKI rows")
    aki = phenotypes.loc[phenotypes["aki_final"]]
    if aki["aki_duration_class"].isna().any() or len(aki) != aki["aki_duration_class"].value_counts().sum():
        raise AssertionError("AKI duration categories do not cover every AKI row")
    checks.append("PASS: mutually exclusive duration categories cover every incident-AKI row")
    if (phenotypes.loc[phenotypes["rrt_within_7d"], "rrt_first_hours"] > 168).any():
        raise AssertionError("RRT event outside the seven-day window")
    checks.append("PASS: all RRT overlay events fall within 0-168 h after ICU admission")
    if states.duplicated(["stay_id", "charttime", "scr_mg_dl"]).mean() > 0.01:
        raise AssertionError("Unexpectedly high duplicate SCr-state rate")
    checks.append("PASS: measurement-state duplicate rate was below 1%")
    if not set(states["aki_stage_at_measurement"].dropna().unique()).issubset({0, 1, 2, 3}):
        raise AssertionError("Invalid measurement-level stage")
    checks.append("PASS: all measurement-level states are in KDIGO SCr stages 0-3")
    return checks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort, labs = load_inputs()
    states = build_measurement_states(cohort, labs)
    rrt_events = extract_rrt_events(cohort)
    rrt = summarize_rrt(cohort, rrt_events)
    phenotypes = derive_patient_phenotypes(cohort, states, rrt)
    audits = make_audits(phenotypes, states)
    contrasts = bootstrap_crude_contrasts(phenotypes)

    phenotypes.to_csv(OUT / "cohort_v26_strict_aki_severity_recovery.csv", index=False)
    states.to_csv(OUT / "creatinine_measurement_states_v26.csv.gz", index=False, compression="gzip")
    audits["severity"].to_csv(OUT / "audit_v26_severity_summary.csv", index=False)
    audits["trajectories"].to_csv(OUT / "audit_v26_recovery_trajectory_summary.csv", index=False)
    audits["observability"].to_csv(OUT / "audit_v26_observability_summary.csv", index=False)
    audits["outcomes"].to_csv(OUT / "audit_v26_clinical_outcomes_by_phenotype.csv", index=False)
    audits["daily"].to_csv(OUT / "audit_v26_daily_state_distribution.csv", index=False)
    audits["transitions"].to_csv(OUT / "audit_v26_observed_transition_matrix.csv", index=False)
    contrasts.to_csv(OUT / "audit_v26_crude_outcome_contrasts_bootstrap.csv", index=False)

    setup_style()
    make_main_figure(phenotypes, audits)
    make_daily_figure(audits)
    write_reports(phenotypes, audits)
    checks = validate_outputs(phenotypes, states)
    report = "# v26 internal validation\n\n" + "\n".join(f"- {check}" for check in checks) + "\n"
    (OUT / "audit_v26_validation_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote v26 severity/trajectory analysis to {OUT}")


if __name__ == "__main__":
    main()
