"""Develop secondary models for severe AKI and early renal trajectories.

The primary study predicts any incident SCr-AKI.  This post-lock secondary
analysis asks two different clinical questions:

1. At ICU admission, 6 h, and 24 h, who will newly develop severe SCr AKI
   (KDIGO stage 2/3) by day 7?  Patients with prior stage 1 AKI remain at risk;
   only patients already at stage 2/3 are removed at positive landmarks.
2. At the first observed AKI measurement, which episodes will persist beyond
   48 h or remain unrecovered at the end of the seven-day/discharge window?

All predictors are restricted to information available at the applicable
landmark.  A single deterministic subject-level 80/20 split is optimized
jointly for the three secondary outcomes and reused throughout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from develop_models_v5 import RANDOM_STATE, threshold_metrics, youden_threshold  # noqa: E402
from dynamic_datasets_v4 import (  # noqa: E402
    STATIC_PREDICTOR_ALLOWLIST,
    aggregate_last,
    aggregate_window,
    baseline_features_at_landmark,
)
from final_sensitivity_and_actionable_analysis_v14 import (  # noqa: E402
    metrics,
    model_pipeline,
    simplified_predictors,
)
from recalibration_and_measurement_intensity_v13 import identify_types_for_columns  # noqa: E402


V26_DIR = ROOT / "outputs" / "modeling_v26_aki_severity_trajectories"
COHORT_FILE = V26_DIR / "cohort_v26_strict_aki_severity_recovery.csv"
STATES_FILE = V26_DIR / "creatinine_measurement_states_v26.csv.gz"
LAB_CACHE = ROOT / "outputs" / "cache_v4" / "aligned_dynamic_labs_pre24_to_post24.csv.gz"
VITAL_CACHE = ROOT / "outputs" / "cache_v4" / "aligned_dynamic_vitals_post24.csv.gz"
MODEL0_FILE = ROOT / "outputs" / "modeling_v4_1" / "modeling_v4_1_0h.csv"
OUT = ROOT / "outputs" / "modeling_v27_severity_recovery"

LANDMARKS = [0, 6, 24]
MODELS = ["Logistic Regression", "XGBoost"]
SELECTED_MODEL = {0: "XGBoost", 6: "XGBoost", 24: "Logistic Regression"}
BOOTSTRAPS = 1000
KEYS = ["subject_id", "hadm_id", "stay_id"]

SEVERE_SCR_OUTCOME = "outcome_severe_scr_after_landmark_to_7d"
SEVERE_SCR_RRT_OUTCOME = "outcome_severe_scr_or_rrt_after_landmark_to_7d"
PERSISTENCE_OUTCOME = "outcome_persistent_aki_beyond_48h"
NONRECOVERY_OUTCOME = "outcome_not_recovered_at_end"

MODEL_COLORS = {"Logistic Regression": "#4C78A8", "XGBoost": "#D97706"}
LANDMARK_COLORS = {0: "#4C78A8", 6: "#7B61A8", 24: "#D97706"}


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
            "legend.fontsize": 6.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
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
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(COHORT_FILE, low_memory=False)
    states = pd.read_csv(STATES_FILE, low_memory=False, parse_dates=["charttime"])
    labs = pd.read_csv(LAB_CACHE, low_memory=False, parse_dates=["charttime"])
    vitals = pd.read_csv(VITAL_CACHE, low_memory=False, parse_dates=["charttime"])
    for column in [
        "intime",
        "index_surgery_date",
        "baseline_scr_time",
        "aki_onset_time_final",
        "rrt_first_time",
        "end_observation_time",
    ]:
        if column in cohort:
            cohort[column] = pd.to_datetime(cohort[column], errors="coerce")
    cohort["aki_final"] = bool_mask(cohort["aki_final"])
    if cohort["stay_id"].duplicated().any():
        raise ValueError("v26 cohort contains duplicate stay_id")
    if set(cohort["stay_id"]) != set(states["stay_id"].unique()):
        raise ValueError("Measurement states do not cover the exact v26 cohort")
    return cohort, states, labs, vitals


def derive_severe_onsets(cohort: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    severe = states.loc[states["aki_stage_at_measurement"].ge(2)].sort_values(
        ["stay_id", "charttime"]
    )
    first = severe.groupby("stay_id", as_index=False).first()[
        ["stay_id", "charttime", "hours_from_icu", "aki_stage_at_measurement"]
    ]
    first = first.rename(
        columns={
            "charttime": "severe_scr_onset_time",
            "hours_from_icu": "severe_scr_onset_hours",
            "aki_stage_at_measurement": "severe_scr_onset_stage",
        }
    )
    result = cohort[["stay_id", "rrt_first_hours", "rrt_within_7d"]].merge(
        first, on="stay_id", how="left", validate="one_to_one"
    )
    result["severe_scr_event"] = result["severe_scr_onset_hours"].notna()
    result["severe_scr_or_rrt_onset_hours"] = result["severe_scr_onset_hours"]
    rrt_hours = pd.to_numeric(result["rrt_first_hours"], errors="coerce")
    result["severe_scr_or_rrt_onset_hours"] = result[
        ["severe_scr_or_rrt_onset_hours"]
    ].assign(rrt=rrt_hours).min(axis=1, skipna=True)
    result["severe_scr_or_rrt_event"] = result["severe_scr_or_rrt_onset_hours"].notna()
    expected = cohort.set_index("stay_id")["maximum_active_scr_stage_7d"].ge(2)
    observed = result.set_index("stay_id")["severe_scr_event"]
    if not expected.equals(observed):
        mismatch = int((expected != observed).sum())
        raise AssertionError(f"Severe SCr onset mismatch with locked peak stage: {mismatch}")
    return result


def current_status_features(
    cohort: pd.DataFrame, states: pd.DataFrame, severe_onsets: pd.DataFrame, landmark: int
) -> pd.DataFrame:
    result = cohort[["stay_id", "aki_final", "aki_onset_hours_final"]].merge(
        severe_onsets[["stay_id", "severe_scr_onset_hours"]], on="stay_id", validate="one_to_one"
    )
    at_landmark = states.loc[states["hours_from_icu"].le(landmark)].sort_values(
        ["stay_id", "charttime"]
    )
    last = at_landmark.groupby("stay_id", as_index=False).tail(1)[
        ["stay_id", "aki_stage_at_measurement", "scr_mg_dl", "scr_ratio_to_baseline"]
    ].rename(
        columns={
            "aki_stage_at_measurement": "current_scr_stage_at_landmark",
            "scr_mg_dl": "current_scr_at_landmark",
            "scr_ratio_to_baseline": "current_scr_ratio_at_landmark",
        }
    )
    result = result.merge(last, on="stay_id", how="left", validate="one_to_one")
    onset = pd.to_numeric(result["aki_onset_hours_final"], errors="coerce")
    severe_onset = pd.to_numeric(result["severe_scr_onset_hours"], errors="coerce")
    result["prior_stage1_aki_by_landmark"] = (
        result["aki_final"].astype(bool)
        & onset.le(landmark)
        & (severe_onset.isna() | severe_onset.gt(landmark))
    )
    result["hours_since_first_aki_at_landmark"] = (landmark - onset).where(
        result["prior_stage1_aki_by_landmark"]
    )
    result["scr_measurement_n_by_landmark"] = (
        states.loc[states["hours_from_icu"].le(landmark)].groupby("stay_id").size()
        .reindex(result["stay_id"])
        .fillna(0)
        .to_numpy()
    )
    return result[
        [
            "stay_id",
            "prior_stage1_aki_by_landmark",
            "hours_since_first_aki_at_landmark",
            "current_scr_stage_at_landmark",
            "current_scr_at_landmark",
            "current_scr_ratio_at_landmark",
            "scr_measurement_n_by_landmark",
        ]
    ]


def build_severe_datasets(
    cohort: pd.DataFrame,
    states: pd.DataFrame,
    labs: pd.DataFrame,
    vitals: pd.DataFrame,
    severe_onsets: pd.DataFrame,
) -> tuple[dict[int, pd.DataFrame], dict[int, list[str]], pd.DataFrame]:
    preindex_labs = aggregate_last(
        labs.loc[labs["hours_from_icu"].between(-24, 0, inclusive="both")],
        "lab_pre24h",
    )
    datasets: dict[int, pd.DataFrame] = {}
    predictor_sets: dict[int, list[str]] = {}
    audit_rows = []
    onset = severe_onsets.set_index("stay_id")
    for landmark in LANDMARKS:
        severe_hours = cohort["stay_id"].map(onset["severe_scr_onset_hours"])
        combined_hours = cohort["stay_id"].map(onset["severe_scr_or_rrt_onset_hours"])
        already_severe = severe_hours.notna() & severe_hours.le(landmark)
        eligible = cohort.loc[~already_severe].copy()

        metadata = eligible[KEYS + ["intime", "index_surgery_date"]].copy()
        metadata["landmark_hours"] = landmark
        predictors = eligible[
            [column for column in STATIC_PREDICTOR_ALLOWLIST if column in eligible.columns]
        ].copy()
        predictors.insert(0, "stay_id", eligible["stay_id"].to_numpy())
        predictors = predictors.merge(
            baseline_features_at_landmark(eligible, landmark),
            on="stay_id",
            how="left",
            validate="one_to_one",
        )
        predictors = predictors.merge(preindex_labs, on="stay_id", how="left", validate="one_to_one")
        if landmark > 0:
            predictors = predictors.merge(
                aggregate_window(labs, landmark, "lab"),
                on="stay_id",
                how="left",
                validate="one_to_one",
            ).merge(
                aggregate_window(vitals, landmark, "vital"),
                on="stay_id",
                how="left",
                validate="one_to_one",
            )
        status = current_status_features(cohort, states, severe_onsets, landmark)
        predictors = predictors.merge(status, on="stay_id", how="left", validate="one_to_one")

        outcome = eligible[["stay_id"]].copy()
        eligible_severe_hours = eligible["stay_id"].map(onset["severe_scr_onset_hours"])
        eligible_combined_hours = eligible["stay_id"].map(onset["severe_scr_or_rrt_onset_hours"])
        outcome[SEVERE_SCR_OUTCOME] = eligible_severe_hours.gt(landmark).fillna(False).astype(int)
        outcome[SEVERE_SCR_RRT_OUTCOME] = eligible_combined_hours.gt(landmark).fillna(False).astype(int)
        outcome["outcome_severe_scr_onset_hours"] = eligible_severe_hours.to_numpy()
        outcome["outcome_severe_scr_or_rrt_onset_hours"] = eligible_combined_hours.to_numpy()
        data = metadata.merge(predictors, on="stay_id", validate="one_to_one").merge(
            outcome, on="stay_id", validate="one_to_one"
        )

        locked = [p for p in simplified_predictors(landmark) if p in data.columns]
        additions = [
            "prior_stage1_aki_by_landmark",
            "hours_since_first_aki_at_landmark",
            "current_scr_stage_at_landmark",
            "current_scr_at_landmark",
            "current_scr_ratio_at_landmark",
            "scr_measurement_n_by_landmark",
        ]
        selected = locked + [p for p in additions if p in data.columns and p not in locked]
        if landmark == 0:
            # At ICU admission the post-index current-state fields carry no
            # information and are omitted rather than imputed as pseudo-data.
            selected = locked
        datasets[landmark] = data
        predictor_sets[landmark] = selected
        progressors = data[SEVERE_SCR_OUTCOME].eq(1) & bool_mask(data["prior_stage1_aki_by_landmark"])
        audit_rows.append(
            {
                "landmark_hours": landmark,
                "risk_set_n": len(data),
                "already_severe_excluded_n": int(already_severe.sum()),
                "severe_scr_event_n": int(data[SEVERE_SCR_OUTCOME].sum()),
                "severe_scr_event_percent": 100 * data[SEVERE_SCR_OUTCOME].mean(),
                "severe_scr_or_rrt_event_n": int(data[SEVERE_SCR_RRT_OUTCOME].sum()),
                "severe_scr_or_rrt_event_percent": 100 * data[SEVERE_SCR_RRT_OUTCOME].mean(),
                "prior_stage1_in_risk_set_n": int(bool_mask(data["prior_stage1_aki_by_landmark"]).sum()),
                "severe_progression_from_prior_stage1_n": int(progressors.sum()),
                "predictor_n": len(selected),
            }
        )
    return datasets, predictor_sets, pd.DataFrame(audit_rows)


def scr_slope(group: pd.DataFrame) -> float:
    if len(group) < 2 or group["hours_from_icu"].nunique() < 2:
        return np.nan
    x = group["hours_from_icu"].to_numpy(float)
    y = group["scr_mg_dl"].to_numpy(float)
    return float(np.polyfit(x, y, 1)[0])


def build_onset_dataset(cohort: pd.DataFrame, states: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    base = pd.read_csv(MODEL0_FILE, low_memory=False)
    base_predictors = [
        p
        for p in simplified_predictors(0)
        if p in base.columns
        and p
        not in {
            "baseline_scr_at_landmark",
            "baseline_scr_source_at_landmark",
            "baseline_to_icu_hours_at_landmark",
        }
    ]
    aki = cohort.loc[cohort["aki_final"]].copy()
    rows = []
    state_groups = {stay: group.sort_values("charttime") for stay, group in states.groupby("stay_id")}
    for row in aki.itertuples(index=False):
        onset = row.aki_onset_time_final
        history = state_groups[row.stay_id]
        history = history.loc[history["charttime"].le(onset)].sort_values("charttime")
        if history.empty:
            raise AssertionError(f"No SCr history at AKI onset for stay {row.stay_id}")
        onset_row = history.iloc[-1]
        recent12 = history.loc[history["charttime"].ge(onset - pd.Timedelta(hours=12))]
        recent24 = history.loc[history["charttime"].ge(onset - pd.Timedelta(hours=24))]
        prior = history.iloc[:-1]
        rows.append(
            {
                "stay_id": row.stay_id,
                "aki_onset_hours_from_icu": row.aki_onset_hours_final,
                "onset_scr_mg_dl": onset_row["scr_mg_dl"],
                "onset_scr_ratio_to_baseline": onset_row["scr_ratio_to_baseline"],
                "onset_scr_delta_from_baseline": onset_row["scr_delta_from_baseline"],
                "onset_aki_stage": onset_row["aki_stage_at_measurement"],
                "onset_absolute_0_3_criterion": onset_row["aki_absolute_0_3_within_prior_48h"],
                "baseline_scr_at_onset": row.baseline_scr_final,
                "baseline_scr_source_at_onset": row.baseline_scr_source,
                "baseline_to_onset_hours": (
                    (onset - row.baseline_scr_time).total_seconds() / 3600
                    if pd.notna(row.baseline_scr_time) and row.baseline_scr_time <= onset
                    else np.nan
                ),
                "scr_n_icu_to_onset": len(history),
                "scr_min_icu_to_onset": history["scr_mg_dl"].min(),
                "scr_max_icu_to_onset": history["scr_mg_dl"].max(),
                "scr_first_icu_to_onset": history.iloc[0]["scr_mg_dl"],
                "scr_change_first_to_onset": onset_row["scr_mg_dl"] - history.iloc[0]["scr_mg_dl"],
                "scr_slope_icu_to_onset_mg_dl_per_h": scr_slope(history),
                "scr_n_prior_to_onset": len(prior),
                "hours_since_previous_scr": (
                    (onset_row["charttime"] - prior.iloc[-1]["charttime"]).total_seconds() / 3600
                    if len(prior)
                    else np.nan
                ),
                "scr_12h_n": len(recent12),
                "scr_12h_min": recent12["scr_mg_dl"].min(),
                "scr_12h_max": recent12["scr_mg_dl"].max(),
                "scr_12h_slope": scr_slope(recent12),
                "scr_24h_n": len(recent24),
                "scr_24h_min": recent24["scr_mg_dl"].min(),
                "scr_24h_max": recent24["scr_mg_dl"].max(),
                "scr_24h_slope": scr_slope(recent24),
            }
        )
    onset_features = pd.DataFrame(rows)
    data = aki[KEYS].merge(
        base[KEYS + base_predictors], on=KEYS, how="left", validate="one_to_one"
    ).merge(onset_features, on="stay_id", how="left", validate="one_to_one")
    data = data.merge(
        aki[
            [
                "stay_id",
                "persistence_evaluable",
                "persistent_aki_scr",
                "end_recovery_evaluable",
                "recovered_at_end_by_kdigo_scr",
                "renal_trajectory_group",
            ]
        ],
        on="stay_id",
        validate="one_to_one",
    )
    data[PERSISTENCE_OUTCOME] = bool_mask(data["persistent_aki_scr"]).astype(int)
    data[NONRECOVERY_OUTCOME] = (~bool_mask(data["recovered_at_end_by_kdigo_scr"])).astype(int)
    onset_predictors = base_predictors + [
        c
        for c in onset_features.columns
        if c != "stay_id"
    ]
    return data, onset_predictors


def choose_joint_subject_split(
    cohort: pd.DataFrame, severe0: pd.DataFrame, onset: pd.DataFrame
) -> tuple[set[int], set[int], pd.DataFrame]:
    groups = cohort["subject_id"].astype(int)
    severe_map = severe0.set_index("stay_id")[SEVERE_SCR_OUTCOME]
    severe_y = cohort["stay_id"].map(severe_map).astype(int)
    persistence = onset.loc[bool_mask(onset["persistence_evaluable"]), ["subject_id", PERSISTENCE_OUTCOME]]
    nonrecovery = onset.loc[bool_mask(onset["end_recovery_evaluable"]), ["subject_id", NONRECOVERY_OUTCOME]]
    splitter = GroupShuffleSplit(n_splits=1000, test_size=0.20, random_state=RANDOM_STATE + 27)
    best = None
    for train_idx, test_idx in splitter.split(cohort, severe_y, groups):
        train_subjects = set(groups.iloc[train_idx])
        test_subjects = set(groups.iloc[test_idx])
        score = 2 * abs(len(test_idx) / len(cohort) - 0.20)
        severe_split_table = cohort[["subject_id"]].copy()
        severe_split_table["_y"] = severe_y.to_numpy()
        for table, outcome in [
            (severe_split_table, "_y"),
            (persistence, PERSISTENCE_OUTCOME),
            (nonrecovery, NONRECOVERY_OUTCOME),
        ]:
            train_values = table.loc[table["subject_id"].isin(train_subjects), outcome]
            test_values = table.loc[table["subject_id"].isin(test_subjects), outcome]
            if train_values.empty or test_values.empty:
                score += 10
                continue
            score += abs(train_values.mean() - table[outcome].mean())
            score += abs(test_values.mean() - table[outcome].mean())
        if best is None or score < best[0]:
            best = (score, train_subjects, test_subjects)
    assert best is not None
    _, train_subjects, test_subjects = best
    if train_subjects & test_subjects:
        raise AssertionError("Subject overlap in joint split")
    assignment = cohort[KEYS].copy()
    assignment["split"] = np.where(
        assignment["subject_id"].isin(test_subjects), "test", "train"
    )
    return train_subjects, test_subjects, assignment


def bootstrap_test_metrics(
    y: np.ndarray, p: np.ndarray, subjects: np.ndarray, seed: int
) -> dict[str, tuple[float, float, int]]:
    unique, inverse = np.unique(subjects, return_inverse=True)
    rows = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    draws = {"auroc": [], "auprc": [], "brier_score": []}
    for _ in range(BOOTSTRAPS):
        sampled = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([rows[i] for i in sampled])
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        draws["auroc"].append(roc_auc_score(yb, pb))
        draws["auprc"].append(average_precision_score(yb, pb))
        draws["brier_score"].append(brier_score_loss(yb, pb))
    result = {}
    for metric, values in draws.items():
        array = np.asarray(values, dtype=float)
        result[metric] = (
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
            len(array),
        )
    return result


def fit_task(
    data: pd.DataFrame,
    predictors: list[str],
    outcome: str,
    task: str,
    landmark: int | str,
    train_subjects: set[int],
    test_subjects: set[int],
    model_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data.loc[data["subject_id"].isin(train_subjects)].copy()
    test = data.loc[data["subject_id"].isin(test_subjects)].copy()
    y_train = train[outcome].astype(int).to_numpy()
    y_test = test[outcome].astype(int).to_numpy()
    continuous, binary, categorical = identify_types_for_columns(data, predictors)
    performance_rows = []
    prediction = test[KEYS].copy()
    prediction["task"] = task
    prediction["landmark"] = landmark
    prediction["outcome"] = y_test
    for model_index, model_name in enumerate(model_names):
        pipeline = model_pipeline(model_name, continuous, binary, categorical)
        pipeline.fit(train[predictors], y_train)
        train_prob = pipeline.predict_proba(train[predictors])[:, 1]
        test_prob = pipeline.predict_proba(test[predictors])[:, 1]
        observed = metrics(y_test, test_prob)
        ci = bootstrap_test_metrics(
            y_test,
            test_prob,
            test["subject_id"].astype(int).to_numpy(),
            RANDOM_STATE + model_index + int(landmark if isinstance(landmark, int) else 50),
        )
        threshold, _ = youden_threshold(y_train, train_prob)
        tm = threshold_metrics(y_test, test_prob, threshold)
        row = {
            "task": task,
            "landmark": landmark,
            "outcome": outcome,
            "model": model_name,
            "predictor_n": len(predictors),
            "train_n": len(train),
            "train_event_n": int(y_train.sum()),
            "train_event_percent": 100 * y_train.mean(),
            "test_n": len(test),
            "test_event_n": int(y_test.sum()),
            "test_event_percent": 100 * y_test.mean(),
            **observed,
            "auroc_ci95_low": ci["auroc"][0],
            "auroc_ci95_high": ci["auroc"][1],
            "auprc_ci95_low": ci["auprc"][0],
            "auprc_ci95_high": ci["auprc"][1],
            "brier_ci95_low": ci["brier_score"][0],
            "brier_ci95_high": ci["brier_score"][1],
            "bootstrap_successful_n": ci["auroc"][2],
            "youden_threshold_training": threshold,
        }
        for key, value in tm.items():
            row[f"youden_test_{key}"] = value
        performance_rows.append(row)
        prediction[f"probability_{model_name.lower().replace(' ', '_')}"] = test_prob
    return pd.DataFrame(performance_rows), prediction


def candidate_missingness(
    severe_sets: dict[int, pd.DataFrame], severe_predictors: dict[int, list[str]], onset: pd.DataFrame, onset_predictors: list[str]
) -> pd.DataFrame:
    rows = []
    for landmark in LANDMARKS:
        for predictor in severe_predictors[landmark]:
            rows.append(
                {
                    "dataset": f"severe_{landmark}h",
                    "predictor": predictor,
                    "n": len(severe_sets[landmark]),
                    "missing_n": int(severe_sets[landmark][predictor].isna().sum()),
                    "missing_percent": 100 * severe_sets[landmark][predictor].isna().mean(),
                }
            )
    for predictor in onset_predictors:
        rows.append(
            {
                "dataset": "aki_onset",
                "predictor": predictor,
                "n": len(onset),
                "missing_n": int(onset[predictor].isna().sum()),
                "missing_percent": 100 * onset[predictor].isna().mean(),
            }
        )
    return pd.DataFrame(rows)


def make_severe_figure(performance: pd.DataFrame, predictions: pd.DataFrame) -> None:
    primary = performance.loc[performance["task"].eq("severe_scr")].copy()
    primary = primary.loc[
        primary.apply(
            lambda r: r["model"] == SELECTED_MODEL[int(r["landmark"])], axis=1
        )
    ].sort_values("landmark")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    ax_a, ax_b, ax_c = axes
    y = np.arange(3)
    auroc = primary["auroc"].to_numpy()
    lower = auroc - primary["auroc_ci95_low"].to_numpy()
    upper = primary["auroc_ci95_high"].to_numpy() - auroc
    ax_a.errorbar(auroc, y, xerr=[lower, upper], fmt="o", color="#344054", capsize=2.5)
    ax_a.set_yticks(y, [f"{int(h)} h" for h in primary["landmark"]])
    ax_a.invert_yaxis()
    ax_a.set_xlim(0.5, 0.9)
    ax_a.axvline(0.5, color="#98A2B3", linestyle="--", linewidth=0.8)
    ax_a.set_xlabel("AUROC (95% CI)")
    ax_a.set_title("Severe SCr-AKI discrimination", loc="left", fontweight="bold")
    panel_label(ax_a, "a")

    for landmark in LANDMARKS:
        model = SELECTED_MODEL[landmark]
        subset = predictions.loc[
            predictions["task"].eq("severe_scr") & predictions["landmark"].astype(str).eq(str(landmark))
        ]
        p = subset[f"probability_{model.lower().replace(' ', '_')}"]
        fpr, tpr, _ = roc_curve(subset["outcome"], p)
        row = primary.loc[primary["landmark"].astype(int).eq(landmark)].iloc[0]
        ax_b.plot(fpr, tpr, color=LANDMARK_COLORS[landmark], linewidth=1.5, label=f"{landmark} h: {row.auroc:.3f}")
        frac, mean = calibration_curve(subset["outcome"], p, n_bins=8, strategy="quantile")
        ax_c.plot(mean, frac, marker="o", markersize=3, color=LANDMARK_COLORS[landmark], linewidth=1.2, label=f"{landmark} h")
    ax_b.plot([0, 1], [0, 1], color="#98A2B3", linestyle="--", linewidth=0.8)
    ax_b.set_xlabel("1 - specificity")
    ax_b.set_ylabel("Sensitivity")
    ax_b.set_title("Held-out ROC curves", loc="left", fontweight="bold")
    ax_b.legend(loc="lower right")
    panel_label(ax_b, "b")
    ax_c.plot([0, 1], [0, 1], color="#98A2B3", linestyle="--", linewidth=0.8)
    ax_c.set_xlim(0, 0.35)
    ax_c.set_ylim(0, 0.35)
    ax_c.set_xlabel("Mean predicted risk")
    ax_c.set_ylabel("Observed risk")
    ax_c.set_title("Calibration", loc="left", fontweight="bold")
    ax_c.legend(loc="upper left")
    panel_label(ax_c, "c")
    save_figure(fig, "figure_v27_severe_aki_dynamic_models")


def make_onset_figure(performance: pd.DataFrame, predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    tasks = [
        ("persistent_aki", "Persistent AKI beyond 48 h"),
        ("nonrecovery", "Not recovered at observation end"),
    ]
    for ax, (task, title) in zip(axes, tasks):
        subset = predictions.loc[predictions["task"].eq(task)]
        rows = performance.loc[performance["task"].eq(task)]
        for model in MODELS:
            p = subset[f"probability_{model.lower().replace(' ', '_')}"]
            fpr, tpr, _ = roc_curve(subset["outcome"], p)
            row = rows.loc[rows["model"].eq(model)].iloc[0]
            ax.plot(
                fpr,
                tpr,
                color=MODEL_COLORS[model],
                linewidth=1.5,
                label=f"{model}: {row.auroc:.3f} ({row.auroc_ci95_low:.3f}-{row.auroc_ci95_high:.3f})",
            )
        ax.plot([0, 1], [0, 1], color="#98A2B3", linestyle="--", linewidth=0.8)
        ax.set_xlabel("1 - specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.legend(loc="lower right")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save_figure(fig, "figure_v27_onset_anchored_trajectory_models")


def write_reports(
    risk_audit: pd.DataFrame,
    onset: pd.DataFrame,
    performance: pd.DataFrame,
    severe_predictors: dict[int, list[str]],
    onset_predictors: list[str],
) -> None:
    selected = performance.loc[
        performance.apply(
            lambda r: (
                (r["task"] == "severe_scr" and r["model"] == SELECTED_MODEL[int(r["landmark"])])
                or r["task"] in {"persistent_aki", "nonrecovery"}
            ),
            axis=1,
        )
    ]
    lines = ["# v27 severe AKI and recovery-trajectory modeling", "", "## Key held-out results", ""]
    for row in selected.itertuples(index=False):
        lines.append(
            f"- **{row.task}, {row.landmark}, {row.model}:** AUROC {row.auroc:.3f} "
            f"(95% CI {row.auroc_ci95_low:.3f}-{row.auroc_ci95_high:.3f}), "
            f"AUPRC {row.auprc:.3f}, Brier {row.brier_score:.3f}; test n={row.test_n:,}, events={row.test_event_n:,}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Severe-AKI models use risk sets that retain patients with stage 1 AKI and remove only those already at stage 2/3. The onset-anchored models use only static, baseline, pre-index, and SCr-kinetic information available at the first AKI-positive measurement. These are secondary internal-validation models and do not replace the locked any-AKI models.",
            "",
            "Persistence and recovery models are conditional on phenotype evaluability. Their performance therefore applies to patients with sufficient subsequent SCr observation, not automatically to every AKI patient.",
        ]
    )
    (OUT / "audit_v27_results_brief.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    readme = f"""# v27 secondary severity and renal-trajectory models

## Severe-AKI risk sets

The 0 h, 6 h, and 24 h outcomes are new KDIGO SCr stage 2/3 events after the landmark through ICU day 7. Patients already at stage 2/3 are excluded at 6 h or 24 h. Prior or current stage 1 AKI is retained and represented with time-restricted status fields. The SCr-or-RRT outcome is a sensitivity target.

## Onset-anchored trajectory models

Time zero is the first SCr-positive AKI measurement. Predictors include the locked 0 h static/pre-index set plus baseline SCr, onset SCr stage/ratio/delta, timing, recent measurement counts, last-measurement interval, and SCr slopes through onset. No post-onset values enter the predictor set.

- Persistence outcome: active SCr AKI beyond 48 h among rows with `persistence_evaluable == True`.
- Nonrecovery outcome: active SCr AKI at the last SCr within 24 h of discharge/day 7 among rows with `end_recovery_evaluable == True`.

## Validation

A single subject-level 80/20 split is reused for every task. Logistic regression and XGBoost use the same preprocessing conventions as the primary project. Confidence intervals use 1,000 subject-cluster bootstrap resamples of the held-out test set.

## Predictor counts

- Severe 0 h: {len(severe_predictors[0])}
- Severe 6 h: {len(severe_predictors[6])}
- Severe 24 h: {len(severe_predictors[24])}
- AKI-onset trajectory: {len(onset_predictors)}

## Scope restrictions

This analysis is post-lock, secondary, and internally validated only. It does not update the locked manuscript, external validation, or deployment artifact. Recovery-model estimates are conditional on follow-up SCr observability.
"""
    (OUT / "audit_v27_readme.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort, states, labs, vitals = load_inputs()
    severe_onsets = derive_severe_onsets(cohort, states)
    severe_sets, severe_predictors, risk_audit = build_severe_datasets(
        cohort, states, labs, vitals, severe_onsets
    )
    onset, onset_predictors = build_onset_dataset(cohort, states)
    train_subjects, test_subjects, assignment = choose_joint_subject_split(
        cohort, severe_sets[0], onset
    )

    performance_tables = []
    prediction_tables = []
    for landmark in LANDMARKS:
        perf, pred = fit_task(
            severe_sets[landmark],
            severe_predictors[landmark],
            SEVERE_SCR_OUTCOME,
            "severe_scr",
            landmark,
            train_subjects,
            test_subjects,
            MODELS,
        )
        performance_tables.append(perf)
        prediction_tables.append(pred)
        perf, pred = fit_task(
            severe_sets[landmark],
            severe_predictors[landmark],
            SEVERE_SCR_RRT_OUTCOME,
            "severe_scr_or_rrt_sensitivity",
            landmark,
            train_subjects,
            test_subjects,
            [SELECTED_MODEL[landmark]],
        )
        performance_tables.append(perf)
        prediction_tables.append(pred)

    persistence_data = onset.loc[bool_mask(onset["persistence_evaluable"])].copy()
    perf, pred = fit_task(
        persistence_data,
        onset_predictors,
        PERSISTENCE_OUTCOME,
        "persistent_aki",
        "AKI onset",
        train_subjects,
        test_subjects,
        MODELS,
    )
    performance_tables.append(perf)
    prediction_tables.append(pred)
    nonrecovery_data = onset.loc[bool_mask(onset["end_recovery_evaluable"])].copy()
    perf, pred = fit_task(
        nonrecovery_data,
        onset_predictors,
        NONRECOVERY_OUTCOME,
        "nonrecovery",
        "AKI onset",
        train_subjects,
        test_subjects,
        MODELS,
    )
    performance_tables.append(perf)
    prediction_tables.append(pred)

    performance = pd.concat(performance_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    missingness = candidate_missingness(severe_sets, severe_predictors, onset, onset_predictors)

    for landmark, data in severe_sets.items():
        data.to_csv(OUT / f"dataset_v27_severe_{landmark}h.csv", index=False)
    onset.to_csv(OUT / "dataset_v27_aki_onset_trajectory.csv", index=False)
    risk_audit.to_csv(OUT / "audit_v27_severe_risk_set_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "target": "persistent AKI beyond 48 h",
                "n": len(persistence_data),
                "event_n": int(persistence_data[PERSISTENCE_OUTCOME].sum()),
                "event_percent": 100 * persistence_data[PERSISTENCE_OUTCOME].mean(),
            },
            {
                "target": "not recovered at observation end",
                "n": len(nonrecovery_data),
                "event_n": int(nonrecovery_data[NONRECOVERY_OUTCOME].sum()),
                "event_percent": 100 * nonrecovery_data[NONRECOVERY_OUTCOME].mean(),
            },
        ]
    ).to_csv(OUT / "audit_v27_onset_target_summary.csv", index=False)
    assignment.to_csv(OUT / "audit_v27_subject_split_assignment.csv", index=False)
    performance.to_csv(OUT / "model_v27_performance_summary.csv", index=False)
    predictions.to_csv(OUT / "model_v27_test_predictions.csv", index=False)
    missingness.to_csv(OUT / "audit_v27_predictor_missingness.csv", index=False)

    setup_style()
    make_severe_figure(performance, predictions)
    make_onset_figure(performance, predictions)
    write_reports(risk_audit, onset, performance, severe_predictors, onset_predictors)
    print(f"Wrote v27 secondary severity/recovery models to {OUT}")


if __name__ == "__main__":
    main()
