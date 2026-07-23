"""v28 temporal and eICU validation of active-episode severe postoperative AKI.

This script validates the v27 secondary severe-AKI target rather than the
locked any-AKI outcome.  Positive-landmark risk sets retain patients with
stage 1 AKI and exclude only active-episode SCr stage 2/3 already documented by
the landmark.  Portable models are fitted in MIMIC-IV and frozen before eICU
evaluation.  Hospital-held-out recalibration and decision-curve analyses are
secondary updates, not replacements for frozen external validation.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from external_recalibration_heterogeneity_v17 import (  # noqa: E402
    expit,
    fit_intercept_only,
    logit,
)
from external_validation_eicu_v16 import (  # noqa: E402
    EICU_ROOT,
    PORTABLE_0H,
    make_model,
    portable_predictors,
)
from final_sensitivity_and_actionable_analysis_v14 import model_pipeline  # noqa: E402
from recalibration_and_measurement_intensity_v13 import (  # noqa: E402
    calibration_intercept_slope,
    identify_types_for_columns,
)


V27 = ROOT / "outputs" / "modeling_v27_severity_recovery"
V16 = ROOT / "outputs" / "modeling_v16_eicu_external_validation"
OUT = ROOT / "outputs" / "modeling_v28_severe_temporal_external"
CACHE = ROOT / "outputs" / "cache_v28"

EICU_COHORT = V16 / "cohort_v16_eicu_external_validation.csv.gz"
EICU_LABS = V16 / "cache" / "eicu_selected_labs.csv.gz"
EICU_TREATMENT = EICU_ROOT / "treatment.csv" / "treatment.csv"

LANDMARKS = [0, 6, 24]
SELECTED_MODEL = {0: "XGBoost", 6: "XGBoost", 24: "Logistic Regression"}
SCR_OUTCOME = "outcome_severe_scr_after_landmark_to_7d"
RRT_OUTCOME = "outcome_severe_scr_or_rrt_after_landmark_to_7d"
EXPECTED_OUTCOME = "outcome_severe_aki"
BOOTSTRAPS = 1000
RANDOM_STATE = 20260728
HORIZON_MIN = 7 * 24 * 60

STATUS_PREDICTORS = [
    "prior_stage1_aki_by_landmark",
    "hours_since_first_aki_at_landmark",
    "current_scr_stage_at_landmark",
    "current_scr_at_landmark",
    "current_scr_ratio_at_landmark",
    "scr_measurement_n_by_landmark",
]

COLORS = {0: "#4C78A8", 6: "#7B61A8", 24: "#D97706"}


def bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).astype(bool)
    return series.astype("string").str.lower().isin(["true", "1", "yes"])


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
    for suffix, kwargs in [
        ("png", {"dpi": 300}),
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": 600}),
    ]:
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold")


def metric_triplet(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    # Preserve the exact saved probabilities for ranking and Brier metrics;
    # clipping is needed only for the calibration logit transform.
    p_for_calibration = np.clip(p, 1e-6, 1 - 1e-6)
    intercept, slope = calibration_intercept_slope(y, p_for_calibration)
    return {
        "n": len(y),
        "event_n": int(y.sum()),
        "event_percent": 100 * float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "mean_predicted_risk": float(p.mean()),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def cluster_bootstrap(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray, seed: int, draws: int = BOOTSTRAPS
) -> dict[str, tuple[float, float, int]]:
    unique, inverse = np.unique(groups, return_inverse=True)
    rows = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    values = {"auroc": [], "auprc": [], "brier_score": []}
    for _ in range(draws):
        sampled = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([rows[i] for i in sampled])
        yb, pb = y[idx], p[idx]
        if np.unique(yb).size < 2:
            continue
        values["auroc"].append(roc_auc_score(yb, pb))
        values["auprc"].append(average_precision_score(yb, pb))
        values["brier_score"].append(brier_score_loss(yb, pb))
    return {
        name: (
            float(np.quantile(v, 0.025)),
            float(np.quantile(v, 0.975)),
            len(v),
        )
        for name, v in values.items()
    }


def derive_eicu_states(
    cohort: pd.DataFrame, labs: pd.DataFrame
) -> pd.DataFrame:
    scr = labs.loc[labs["labname"].eq("creatinine")].copy()
    scr["offset_min"] = pd.to_numeric(scr["offset_min"], errors="coerce")
    scr["value"] = pd.to_numeric(scr["value"], errors="coerce")
    scr = scr.dropna(subset=["offset_min", "value"]).sort_values(["stay_id", "offset_min"])
    baseline = cohort.set_index("stay_id")["baseline_scr_at_landmark"]
    evaluable = set(cohort.loc[bool_mask(cohort["incident_aki_evaluable"]), "stay_id"].astype(int))
    rows = []
    for stay_id, group in scr.loc[scr["stay_id"].isin(evaluable)].groupby("stay_id", sort=False):
        base = float(baseline.loc[stay_id])
        window: deque[tuple[float, float]] = deque()
        for lab in group.loc[group["offset_min"].between(-48 * 60, HORIZON_MIN)].itertuples():
            while window and lab.offset_min - window[0][0] > 48 * 60:
                window.popleft()
            absolute = bool(window) and float(lab.value) - min(v for _, v in window) >= 0.3 - 1e-12
            ratio = float(lab.value) / base
            active = bool(ratio >= 1.5 or absolute)
            if active and (ratio >= 3 or float(lab.value) >= 4):
                stage = 3
            elif active and ratio >= 2:
                stage = 2
            elif active:
                stage = 1
            else:
                stage = 0
            if lab.offset_min > 0:
                rows.append(
                    {
                        "stay_id": int(stay_id),
                        "offset_min": float(lab.offset_min),
                        "hours_from_icu": float(lab.offset_min) / 60,
                        "scr_mg_dl": float(lab.value),
                        "scr_ratio_to_baseline": ratio,
                        "aki_active": active,
                        "aki_stage_at_measurement": stage,
                    }
                )
            window.append((float(lab.offset_min), float(lab.value)))
    states = pd.DataFrame(rows)
    if states.empty:
        raise ValueError("No eICU measurement states derived")
    return states


def extract_eicu_rrt(cohort: pd.DataFrame) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / "eicu_rrt_treatments_v28.csv.gz"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    ids = set(cohort["stay_id"].astype(int))
    include = r"dialysis|hemodial|haemodial|renal replacement|hemofiltration|haemofiltration|cvvh|cvvhd|cvvhdf|crrt|peritoneal dialysis"
    exclude = r"chronic renal failure|insertion|catheter|shunt|access surgery|cannula placement"
    parts = []
    for chunk in pd.read_csv(
        EICU_TREATMENT,
        usecols=["patientunitstayid", "treatmentoffset", "treatmentstring"],
        chunksize=500_000,
        low_memory=False,
    ):
        text = chunk["treatmentstring"].fillna("")
        selected = chunk.loc[
            chunk["patientunitstayid"].isin(ids)
            & text.str.contains(include, case=False, regex=True)
            & ~text.str.contains(exclude, case=False, regex=True)
        ].copy()
        if not selected.empty:
            parts.append(selected)
    events = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not events.empty:
        events = events.rename(
            columns={"patientunitstayid": "stay_id", "treatmentoffset": "rrt_offset_min"}
        )
        events["rrt_offset_min"] = pd.to_numeric(events["rrt_offset_min"], errors="coerce")
        events = events.loc[events["rrt_offset_min"].gt(0) & events["rrt_offset_min"].le(HORIZON_MIN)]
    events.to_csv(cache, index=False, compression="gzip")
    return events


def eicu_onset_table(cohort: pd.DataFrame, states: pd.DataFrame, rrt: pd.DataFrame) -> pd.DataFrame:
    severe = states.loc[states["aki_stage_at_measurement"].ge(2)].sort_values(
        ["stay_id", "offset_min"]
    )
    first = severe.groupby("stay_id", as_index=False).first()[
        ["stay_id", "hours_from_icu", "aki_stage_at_measurement"]
    ].rename(
        columns={
            "hours_from_icu": "severe_scr_onset_hours",
            "aki_stage_at_measurement": "severe_scr_onset_stage",
        }
    )
    result = cohort[["stay_id"]].merge(first, on="stay_id", how="left", validate="one_to_one")
    if rrt.empty:
        rrt_first = pd.Series(dtype=float)
    else:
        rrt_first = rrt.groupby("stay_id")["rrt_offset_min"].min() / 60
    result["rrt_first_hours"] = result["stay_id"].map(rrt_first)
    result["severe_scr_or_rrt_onset_hours"] = result[
        ["severe_scr_onset_hours", "rrt_first_hours"]
    ].min(axis=1, skipna=True)
    return result


def status_at_landmark(
    cohort: pd.DataFrame, states: pd.DataFrame, onsets: pd.DataFrame, landmark: int
) -> pd.DataFrame:
    result = cohort[["stay_id", "aki_onset_hours_final"]].merge(
        onsets[["stay_id", "severe_scr_onset_hours"]], on="stay_id", validate="one_to_one"
    )
    measured = states.loc[states["hours_from_icu"].le(landmark)].sort_values(
        ["stay_id", "offset_min"]
    )
    last = measured.groupby("stay_id", as_index=False).tail(1)[
        ["stay_id", "aki_stage_at_measurement", "scr_mg_dl", "scr_ratio_to_baseline"]
    ].rename(
        columns={
            "aki_stage_at_measurement": "current_scr_stage_at_landmark",
            "scr_mg_dl": "current_scr_at_landmark",
            "scr_ratio_to_baseline": "current_scr_ratio_at_landmark",
        }
    )
    result = result.merge(last, on="stay_id", how="left", validate="one_to_one")
    any_onset = pd.to_numeric(result["aki_onset_hours_final"], errors="coerce")
    severe_onset = pd.to_numeric(result["severe_scr_onset_hours"], errors="coerce")
    result["prior_stage1_aki_by_landmark"] = (
        any_onset.le(landmark) & (severe_onset.isna() | severe_onset.gt(landmark))
    )
    result["hours_since_first_aki_at_landmark"] = (landmark - any_onset).where(
        result["prior_stage1_aki_by_landmark"]
    )
    counts = measured.groupby("stay_id").size()
    result["scr_measurement_n_by_landmark"] = result["stay_id"].map(counts).fillna(0)
    return result[["stay_id", *STATUS_PREDICTORS]]


def build_eicu_landmarks(
    cohort: pd.DataFrame, states: pd.DataFrame, onsets: pd.DataFrame
) -> dict[int, pd.DataFrame]:
    evaluable = cohort.loc[bool_mask(cohort["incident_aki_evaluable"])].copy()
    onset = onsets.set_index("stay_id")
    outputs = {}
    for landmark in LANDMARKS:
        severe_hours = evaluable["stay_id"].map(onset["severe_scr_onset_hours"])
        combined_hours = evaluable["stay_id"].map(onset["severe_scr_or_rrt_onset_hours"])
        eligible = evaluable.loc[severe_hours.isna() | severe_hours.gt(landmark)].copy()
        eligible["severe_scr_onset_hours"] = eligible["stay_id"].map(
            onset["severe_scr_onset_hours"]
        )
        eligible["severe_scr_or_rrt_onset_hours"] = eligible["stay_id"].map(
            onset["severe_scr_or_rrt_onset_hours"]
        )
        eligible[SCR_OUTCOME] = eligible["stay_id"].map(onset["severe_scr_onset_hours"]).gt(landmark).fillna(False).astype(int)
        eligible[RRT_OUTCOME] = eligible["stay_id"].map(onset["severe_scr_or_rrt_onset_hours"]).gt(landmark).fillna(False).astype(int)
        eligible = eligible.merge(
            status_at_landmark(evaluable, states, onsets, landmark),
            on="stay_id",
            how="left",
            validate="one_to_one",
        )
        outputs[landmark] = eligible
    return outputs


def portable_set(landmark: int, dev: pd.DataFrame, external: pd.DataFrame) -> list[str]:
    predictors = portable_predictors(landmark)
    if landmark > 0:
        predictors += STATUS_PREDICTORS
    missing = sorted(set(predictors) - set(dev.columns)) + sorted(set(predictors) - set(external.columns))
    if missing:
        raise ValueError(f"Portable predictor missing at {landmark} h: {sorted(set(missing))}")
    return predictors


def fit_external_models(
    external_sets: dict[int, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    perf_rows, prediction_rows, map_rows = [], [], []
    for landmark in LANDMARKS:
        dev_base = pd.read_csv(V27 / f"dataset_v27_severe_{landmark}h.csv", low_memory=False)
        ext_base = external_sets[landmark].copy()
        predictors = portable_set(landmark, dev_base, ext_base)
        continuous, binary, categorical = identify_types_for_columns(dev_base, predictors)
        for target, label in [(SCR_OUTCOME, "SCr stage 2/3"), (RRT_OUTCOME, "SCr stage 2/3 or RRT sensitivity")]:
            dev = dev_base.copy()
            ext = ext_base.copy()
            # Each landmark prediction target requires its own event-free risk
            # set.  In particular, a patient treated with RRT before the
            # landmark is no longer at risk for first SCr-stage-2/3-or-RRT,
            # even when SCr stage 2/3 has not yet appeared.
            if target == RRT_OUTCOME:
                dev_onset = pd.to_numeric(
                    dev["outcome_severe_scr_or_rrt_onset_hours"], errors="coerce"
                )
                ext_onset = pd.to_numeric(
                    ext["severe_scr_or_rrt_onset_hours"], errors="coerce"
                )
                dev = dev.loc[dev_onset.isna() | dev_onset.gt(landmark)].copy()
                ext = ext.loc[ext_onset.isna() | ext_onset.gt(landmark)].copy()
            model = make_model(landmark, continuous, binary, categorical)
            model.fit(dev[predictors], dev[target].astype(int))
            p = model.predict_proba(ext[predictors])[:, 1]
            y = ext[target].astype(int).to_numpy()
            row = {
                "landmark_hours": landmark,
                "target": label,
                "model": SELECTED_MODEL[landmark],
                "predictor_n": len(predictors),
                **metric_triplet(y, p),
            }
            ci = cluster_bootstrap(y, p, ext["subject_id"].astype(str).to_numpy(), RANDOM_STATE + landmark + len(label))
            for metric, (low, high, successful) in ci.items():
                row[f"{metric}_ci95_low"] = low
                row[f"{metric}_ci95_high"] = high
                row["bootstrap_successful_n"] = successful
            perf_rows.append(row)
            out = ext[
                [
                    "stay_id",
                    "subject_id",
                    "hospitalid",
                    "operative_system",
                    "baseline_scr_source",
                    "ckd",
                    "prior_stage1_aki_by_landmark",
                    target,
                ]
            ].copy()
            out = out.rename(columns={target: EXPECTED_OUTCOME})
            out["landmark_hours"] = landmark
            out["target"] = label
            out["predicted_risk_frozen"] = p
            prediction_rows.append(out)
        for predictor in predictors:
            map_rows.append(
                {
                    "landmark_hours": landmark,
                    "predictor": predictor,
                    "mimic_missing_percent": 100 * dev_base[predictor].isna().mean(),
                    "eicu_missing_percent": 100 * ext_base[predictor].isna().mean(),
                }
            )
    return pd.DataFrame(perf_rows), pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(map_rows)


def rolling_temporal_validation() -> pd.DataFrame:
    missingness = pd.read_csv(V27 / "audit_v27_predictor_missingness.csv")
    rows = []
    for landmark in LANDMARKS:
        data = pd.read_csv(V27 / f"dataset_v27_severe_{landmark}h.csv", low_memory=False)
        data["intime"] = pd.to_datetime(data["intime"], errors="coerce")
        data["icu_year"] = data["intime"].dt.year.astype(int)
        predictors = missingness.loc[
            missingness["dataset"].eq(f"severe_{landmark}h"), "predictor"
        ].tolist()
        years = np.sort(data["icu_year"].unique())
        blocks = [block for block in np.array_split(years, 5) if len(block)]
        for block_index, block in enumerate(blocks[1:], start=1):
            start, end = int(block.min()), int(block.max())
            test = data.loc[data["icu_year"].between(start, end)].copy()
            train = data.loc[data["icu_year"].lt(start)].copy()
            train = train.loc[~train["subject_id"].isin(set(test["subject_id"]))].copy()
            if len(train) < 1000 or len(test) < 300 or train[SCR_OUTCOME].nunique() < 2 or test[SCR_OUTCOME].nunique() < 2:
                continue
            continuous, binary, categorical = identify_types_for_columns(train, predictors)
            model = model_pipeline(SELECTED_MODEL[landmark], continuous, binary, categorical)
            model.fit(train[predictors], train[SCR_OUTCOME].astype(int))
            p = model.predict_proba(test[predictors])[:, 1]
            rows.append(
                {
                    "landmark_hours": landmark,
                    "test_year_start": start,
                    "test_year_end": end,
                    "train_n": len(train),
                    "test_n": len(test),
                    "train_event_n": int(train[SCR_OUTCOME].sum()),
                    "test_event_n": int(test[SCR_OUTCOME].sum()),
                    "model": SELECTED_MODEL[landmark],
                    **metric_triplet(test[SCR_OUTCOME].to_numpy(), p),
                }
            )
    return pd.DataFrame(rows)


def select_hospital_split(predictions: pd.DataFrame) -> tuple[set[int], set[int], pd.DataFrame]:
    base = predictions.loc[
        predictions["target"].eq("SCr stage 2/3") & predictions["landmark_hours"].eq(0)
    ].copy()
    groups = pd.to_numeric(base["hospitalid"], errors="coerce").astype(int)
    y = base[EXPECTED_OUTCOME].astype(int)
    splitter = GroupShuffleSplit(n_splits=1000, test_size=0.20, random_state=RANDOM_STATE)
    best = None
    for cal_idx, test_idx in splitter.split(base, y, groups):
        score = abs(len(test_idx) / len(base) - 0.2) + abs(y.iloc[test_idx].mean() - y.mean())
        if best is None or score < best[0]:
            best = (score, cal_idx, test_idx)
    assert best is not None
    _, cal_idx, test_idx = best
    cal_hospitals = set(groups.iloc[cal_idx])
    test_hospitals = set(groups.iloc[test_idx])
    if cal_hospitals & test_hospitals:
        raise AssertionError("Hospital overlap")
    audit = pd.DataFrame(
        [
            {
                "partition": "recalibration hospitals",
                "hospital_n": len(cal_hospitals),
                "n_0h": len(cal_idx),
                "event_percent_0h": 100 * y.iloc[cal_idx].mean(),
            },
            {
                "partition": "held-out hospitals",
                "hospital_n": len(test_hospitals),
                "n_0h": len(test_idx),
                "event_percent_0h": 100 * y.iloc[test_idx].mean(),
            },
        ]
    )
    return cal_hospitals, test_hospitals, audit


def recalibrate(
    predictions: pd.DataFrame, cal_hospitals: set[int], test_hospitals: set[int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = predictions.loc[predictions["target"].eq("SCr stage 2/3")].copy()
    rows, params, test_outputs = [], [], []
    for landmark in LANDMARKS:
        data = primary.loc[primary["landmark_hours"].eq(landmark)].copy()
        data["hospitalid"] = pd.to_numeric(data["hospitalid"], errors="coerce").astype(int)
        cal = data.loc[data["hospitalid"].isin(cal_hospitals)]
        test = data.loc[data["hospitalid"].isin(test_hospitals)].copy()
        y_cal = cal[EXPECTED_OUTCOME].astype(int).to_numpy()
        p_cal = cal["predicted_risk_frozen"].to_numpy()
        y_test = test[EXPECTED_OUTCOME].astype(int).to_numpy()
        p_test = test["predicted_risk_frozen"].to_numpy()
        offset = fit_intercept_only(y_cal, p_cal)
        platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        platt.fit(logit(p_cal).reshape(-1, 1), y_cal)
        intercept, slope = float(platt.intercept_[0]), float(platt.coef_[0, 0])
        probabilities = {
            "frozen": p_test,
            "intercept update": expit(logit(p_test) + offset),
            "logistic recalibration": expit(intercept + slope * logit(p_test)),
        }
        for method, p in probabilities.items():
            rows.append(
                {
                    "landmark_hours": landmark,
                    "method": method,
                    **metric_triplet(y_test, p),
                }
            )
        params.append(
            {
                "landmark_hours": landmark,
                "recalibration_n": len(cal),
                "recalibration_event_n": int(y_cal.sum()),
                "intercept_only_offset": offset,
                "logistic_recalibration_intercept": intercept,
                "logistic_recalibration_slope": slope,
            }
        )
        test["predicted_risk_intercept_updated"] = probabilities["intercept update"]
        test["predicted_risk_logistic_recalibrated"] = probabilities["logistic recalibration"]
        test_outputs.append(test)
    return pd.DataFrame(rows), pd.DataFrame(params), pd.concat(test_outputs, ignore_index=True)


def hospital_heterogeneity(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = predictions.loc[predictions["target"].eq("SCr stage 2/3")]
    rows, summaries = [], []
    for landmark in LANDMARKS:
        lm = primary.loc[primary["landmark_hours"].eq(landmark)]
        for hospital, group in lm.groupby("hospitalid"):
            y = group[EXPECTED_OUTCOME].astype(int).to_numpy()
            p = group["predicted_risk_frozen"].to_numpy()
            evaluable = len(group) >= 50 and y.sum() >= 5 and (len(y) - y.sum()) >= 20
            row = {
                "landmark_hours": landmark,
                "hospitalid": hospital,
                "n": len(group),
                "event_n": int(y.sum()),
                "event_percent": 100 * y.mean(),
                "auroc_evaluable": evaluable,
            }
            if evaluable:
                row.update(metric_triplet(y, p))
            rows.append(row)
        table = pd.DataFrame([r for r in rows if r["landmark_hours"] == landmark])
        values = table.loc[table["auroc_evaluable"], "auroc"].dropna()
        summaries.append(
            {
                "landmark_hours": landmark,
                "hospital_n": lm["hospitalid"].nunique(),
                "hospital_n_auroc_evaluable": len(values),
                "hospital_auroc_median": values.median() if len(values) else np.nan,
                "hospital_auroc_q1": values.quantile(0.25) if len(values) else np.nan,
                "hospital_auroc_q3": values.quantile(0.75) if len(values) else np.nan,
                "hospital_auroc_min": values.min() if len(values) else np.nan,
                "hospital_auroc_max": values.max() if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def net_benefit(y: np.ndarray, decision: np.ndarray, threshold: float) -> float:
    y = np.asarray(y, dtype=int)
    decision = np.asarray(decision, dtype=bool)
    tp = np.sum(decision & (y == 1))
    fp = np.sum(decision & (y == 0))
    return float(tp / len(y) - fp / len(y) * threshold / (1 - threshold))


def decision_curves(
    internal_predictions: pd.DataFrame, external_heldout: pd.DataFrame
) -> pd.DataFrame:
    # v27 stored the landmark under ``landmark`` whereas the external-v28
    # prediction table uses the more explicit ``landmark_hours``.  Normalize
    # the interface here so DCA cannot silently select an empty risk set.
    internal_predictions = internal_predictions.rename(columns={"landmark": "landmark_hours"})
    rows = []
    thresholds = np.linspace(0.01, 0.20, 20)
    for database, data in [("MIMIC held-out", internal_predictions), ("eICU held-out hospitals", external_heldout)]:
        for landmark in LANDMARKS:
            subset = data.loc[data["landmark_hours"].astype(str).eq(str(landmark))].copy()
            if database.startswith("MIMIC"):
                model = SELECTED_MODEL[landmark]
                probability_columns = {"model": f"probability_{model.lower().replace(' ', '_')}"}
                outcome = "outcome"
            else:
                probability_columns = {
                    "frozen model": "predicted_risk_frozen",
                    "logistic recalibration": "predicted_risk_logistic_recalibrated",
                }
                outcome = EXPECTED_OUTCOME
            y = subset[outcome].astype(int).to_numpy()
            for threshold in thresholds:
                methods = {
                    "treat all": np.ones(len(subset), dtype=bool),
                    "treat none": np.zeros(len(subset), dtype=bool),
                    "prior stage 1 rule": bool_mask(subset["prior_stage1_aki_by_landmark"]).to_numpy(),
                    "CKD rule": bool_mask(subset["ckd"]).to_numpy(),
                }
                for label, column in probability_columns.items():
                    methods[label] = subset[column].to_numpy() >= threshold
                for method, decision in methods.items():
                    rows.append(
                        {
                            "database": database,
                            "landmark_hours": landmark,
                            "threshold": threshold,
                            "method": method,
                            "n": len(subset),
                            "event_n": int(y.sum()),
                            "net_benefit": net_benefit(y, decision, threshold),
                            "alerts_per_100": 100 * decision.mean(),
                        }
                    )
    return pd.DataFrame(rows)


def make_figures(
    rolling: pd.DataFrame,
    external_perf: pd.DataFrame,
    external_predictions: pd.DataFrame,
    recalibration: pd.DataFrame,
    recalibrated_predictions: pd.DataFrame,
    dca: pd.DataFrame,
) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    ax_a, ax_b, ax_c = axes
    for landmark in LANDMARKS:
        subset = rolling.loc[rolling["landmark_hours"].eq(landmark)].sort_values("test_year_start")
        ax_a.plot(
            subset["test_year_start"], subset["auroc"], marker="o", color=COLORS[landmark], label=f"{landmark} h"
        )
    ax_a.axhline(0.5, color="#98A2B3", linestyle="--", linewidth=0.8)
    ax_a.set_xlabel("Test-period start year")
    ax_a.set_ylabel("AUROC")
    ax_a.set_title("Rolling temporal validation", loc="left", fontweight="bold")
    ax_a.legend()
    panel_label(ax_a, "a")

    primary = external_predictions.loc[external_predictions["target"].eq("SCr stage 2/3")]
    for landmark in LANDMARKS:
        subset = primary.loc[primary["landmark_hours"].eq(landmark)]
        y = subset[EXPECTED_OUTCOME]
        p = subset["predicted_risk_frozen"]
        fpr, tpr, _ = roc_curve(y, p)
        auc = external_perf.loc[
            external_perf["target"].eq("SCr stage 2/3")
            & external_perf["landmark_hours"].eq(landmark),
            "auroc",
        ].iloc[0]
        ax_b.plot(fpr, tpr, color=COLORS[landmark], label=f"{landmark} h: {auc:.3f}")
    ax_b.plot([0, 1], [0, 1], color="#98A2B3", linestyle="--", linewidth=0.8)
    ax_b.set_xlabel("1 - specificity")
    ax_b.set_ylabel("Sensitivity")
    ax_b.set_title("Frozen eICU validation", loc="left", fontweight="bold")
    ax_b.legend(loc="lower right")
    panel_label(ax_b, "b")

    for landmark in LANDMARKS:
        subset = recalibrated_predictions.loc[recalibrated_predictions["landmark_hours"].eq(landmark)]
        for method, column, style in [
            ("Frozen", "predicted_risk_frozen", "--"),
            ("Updated", "predicted_risk_logistic_recalibrated", "-"),
        ]:
            frac, mean = calibration_curve(subset[EXPECTED_OUTCOME], subset[column], n_bins=8, strategy="quantile")
            ax_c.plot(mean, frac, color=COLORS[landmark], linestyle=style, marker="o", markersize=2.5, label=f"{landmark} h {method}")
    ax_c.plot([0, 0.35], [0, 0.35], color="#98A2B3", linestyle=":", linewidth=0.8)
    ax_c.set_xlim(0, 0.35)
    ax_c.set_ylim(0, 0.35)
    ax_c.set_xlabel("Mean predicted risk")
    ax_c.set_ylabel("Observed risk")
    ax_c.set_title("Held-out hospital calibration", loc="left", fontweight="bold")
    ax_c.legend(ncol=1, fontsize=5.6)
    panel_label(ax_c, "c")
    save_figure(fig, "figure_v28_severe_temporal_external_validation")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True, sharey=True)
    for ax, landmark in zip(axes, LANDMARKS):
        subset = dca.loc[
            dca["database"].eq("eICU held-out hospitals") & dca["landmark_hours"].eq(landmark)
        ]
        styles = {
            "logistic recalibration": ("#D97706", "-"),
            "frozen model": ("#4C78A8", "-"),
            "prior stage 1 rule": ("#7B61A8", "--"),
            "CKD rule": ("#667085", "--"),
            "treat all": ("#98A2B3", ":"),
        }
        for method, (color, linestyle) in styles.items():
            line = subset.loc[subset["method"].eq(method)]
            ax.plot(line["threshold"], line["net_benefit"], color=color, linestyle=linestyle, label=method)
        ax.axhline(0, color="#101828", linewidth=0.7)
        ax.set_title(f"{landmark} h", fontweight="bold")
        ax.set_xlabel("Risk threshold")
    axes[0].set_ylabel("Net benefit")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=5.4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        frameon=False,
    )
    save_figure(fig, "figure_v28_severe_external_decision_curve")


def write_report(
    rolling: pd.DataFrame,
    external: pd.DataFrame,
    recal: pd.DataFrame,
    hospital_summary: pd.DataFrame,
    split: pd.DataFrame,
) -> None:
    lines = ["# v28 severe-AKI temporal and external validation", "", "## Frozen eICU validation", ""]
    for row in external.loc[external["target"].eq("SCr stage 2/3")].itertuples():
        lines.append(
            f"- {row.landmark_hours} h: n={row.n:,}, events={row.event_n:,} ({row.event_percent:.1f}%); "
            f"AUROC {row.auroc:.3f} (95% CI {row.auroc_ci95_low:.3f}-{row.auroc_ci95_high:.3f}), "
            f"AUPRC {row.auprc:.3f}, Brier {row.brier_score:.3f}, calibration slope {row.calibration_slope:.2f}."
        )
    lines.extend(["", "## Hospital-held-out recalibration", ""])
    for landmark in LANDMARKS:
        subset = recal.loc[recal["landmark_hours"].eq(landmark)]
        frozen = subset.loc[subset["method"].eq("frozen")].iloc[0]
        updated = subset.loc[subset["method"].eq("logistic recalibration")].iloc[0]
        lines.append(
            f"- {landmark} h: held-out hospitals n={int(frozen.n):,}; frozen Brier {frozen.brier_score:.3f}, "
            f"slope {frozen.calibration_slope:.2f}; recalibrated Brier {updated.brier_score:.3f}, slope {updated.calibration_slope:.2f}."
        )
    lines.extend(["", "## Rolling temporal validation", ""])
    for landmark in LANDMARKS:
        subset = rolling.loc[rolling["landmark_hours"].eq(landmark)]
        lines.append(
            f"- {landmark} h: {len(subset)} expanding-window evaluations; AUROC range "
            f"{subset.auroc.min():.3f}-{subset.auroc.max():.3f}."
        )
    lines.extend(["", "## Hospital heterogeneity", ""])
    for row in hospital_summary.itertuples():
        lines.append(
            f"- {row.landmark_hours} h: {row.hospital_n_auroc_evaluable} hospitals met the prespecified minimum; "
            f"median AUROC {row.hospital_auroc_median:.3f} (IQR {row.hospital_auroc_q1:.3f}-{row.hospital_auroc_q3:.3f})."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Frozen external validation is the transportability estimate. Recalibration is a local update learned in separate hospitals and does not improve ranking discrimination. DCA is exploratory and should be interpreted only over clinically plausible severe-AKI thresholds. The eICU RRT target is a treatment-record sensitivity because chronic dialysis and documentation intensity cannot be perfectly separated.",
        ]
    )
    (OUT / "audit_v28_severe_results_brief.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(EICU_COHORT, low_memory=False)
    labs = pd.read_csv(EICU_LABS, low_memory=False)
    states = derive_eicu_states(cohort, labs)
    rrt = extract_eicu_rrt(cohort)
    onsets = eicu_onset_table(cohort, states, rrt)
    external_sets = build_eicu_landmarks(cohort, states, onsets)

    external_perf, external_predictions, mapping = fit_external_models(external_sets)
    rolling = rolling_temporal_validation()
    cal_hospitals, test_hospitals, split = select_hospital_split(external_predictions)
    recalibration, parameters, recalibrated_predictions = recalibrate(
        external_predictions, cal_hospitals, test_hospitals
    )
    hospitals, hospital_summary = hospital_heterogeneity(external_predictions)

    internal_predictions = pd.read_csv(V27 / "model_v27_test_predictions.csv", low_memory=False)
    internal_predictions = internal_predictions.loc[internal_predictions["task"].eq("severe_scr")].copy()
    for landmark in LANDMARKS:
        dev = pd.read_csv(V27 / f"dataset_v27_severe_{landmark}h.csv", usecols=["stay_id", "ckd", "prior_stage1_aki_by_landmark"])
        mask = internal_predictions["landmark"].astype(str).eq(str(landmark))
        internal_predictions.loc[mask, ["ckd", "prior_stage1_aki_by_landmark"]] = internal_predictions.loc[mask, ["stay_id"]].merge(
            dev, on="stay_id", how="left", validate="one_to_one"
        )[["ckd", "prior_stage1_aki_by_landmark"]].to_numpy()
    dca = decision_curves(internal_predictions, recalibrated_predictions)

    states.to_csv(OUT / "eicu_creatinine_measurement_states_v28.csv.gz", index=False, compression="gzip")
    onsets.to_csv(OUT / "eicu_severe_onset_rrt_v28.csv", index=False)
    for landmark, data in external_sets.items():
        data.to_csv(OUT / f"dataset_v28_eicu_severe_{landmark}h.csv.gz", index=False, compression="gzip")
    rolling.to_csv(OUT / "analysis_v28_mimic_rolling_severe_validation.csv", index=False)
    external_perf.to_csv(OUT / "model_v28_eicu_frozen_severe_performance.csv", index=False)
    external_predictions.to_csv(OUT / "model_v28_eicu_frozen_severe_predictions.csv", index=False)
    mapping.to_csv(OUT / "audit_v28_severe_feature_harmonization.csv", index=False)
    split.to_csv(OUT / "audit_v28_hospital_recalibration_split.csv", index=False)
    pd.DataFrame(
        [
            *({"hospitalid": hospital, "partition": "recalibration"} for hospital in sorted(cal_hospitals)),
            *({"hospitalid": hospital, "partition": "held-out validation"} for hospital in sorted(test_hospitals)),
        ]
    ).to_csv(OUT / "audit_v28_hospital_partition_membership.csv", index=False)
    recalibration.to_csv(OUT / "model_v28_heldout_hospital_recalibration_performance.csv", index=False)
    parameters.to_csv(OUT / "model_v28_recalibration_parameters.csv", index=False)
    recalibrated_predictions.to_csv(OUT / "model_v28_heldout_hospital_predictions.csv", index=False)
    hospitals.to_csv(OUT / "analysis_v28_hospital_heterogeneity.csv", index=False)
    hospital_summary.to_csv(OUT / "analysis_v28_hospital_heterogeneity_summary.csv", index=False)
    dca.to_csv(OUT / "analysis_v28_severe_decision_curve.csv", index=False)

    make_figures(
        rolling,
        external_perf,
        external_predictions,
        recalibration,
        recalibrated_predictions,
        dca,
    )
    write_report(rolling, external_perf, recalibration, hospital_summary, split)
    print(f"Wrote v28 severe temporal/external validation to {OUT}")


if __name__ == "__main__":
    main()
