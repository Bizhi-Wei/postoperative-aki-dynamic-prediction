"""v14 final sensitivity analyses and cautious actionable-marker association.

Components:

1. Conservative urine-output AKI sensitivity
   - reports SCr-only, UO-only, and combined SCr-or-UO outcomes at stricter
     urine-output coverage thresholds.

2. Rolling temporal validation
   - expanding-window chronological validation using selected parsimonious
     models: 0h XGBoost, 6h XGBoost, 24h Logistic Regression.

3. Selected parsimonious model finalization
   - locks predictor sets, selected model families, and held-out performance.

4. Secondary association analysis
   - estimates adjusted associations for cautious, clinically actionable
     markers. This is prognostic association, not causal inference.
"""

from __future__ import annotations

from pathlib import Path
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

import xgboost as xgb

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from develop_models_v5 import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    LANDMARKS,
    METADATA,
    OUTCOME,
    RANDOM_STATE,
    choose_grouped_split,
    load_data,
)
from recalibration_and_measurement_intensity_v13 import (  # noqa: E402
    calibration_intercept_slope,
    identify_types_for_columns,
    make_preprocessor_custom,
)


warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v14_final_sensitivities"
V6_FILE = PROJECT_ROOT / "outputs" / "v6_urine_output_or_pacu" / "cohort_v6_strict_primary_aki_scr_uo.csv"
V10_DIR = PROJECT_ROOT / "outputs" / "modeling_v10_simplified_model"
DYNAMIC_DIR = PROJECT_ROOT / "outputs" / "dynamic_v4"

SELECTED_MODELS = {0: "XGBoost", 6: "XGBoost", 24: "Logistic Regression"}
ROLLING_BLOCKS = 5
MIN_ROLLING_TRAIN_N = 1000
MIN_ROLLING_TEST_N = 300

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}


def bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).astype(bool)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def model_pipeline(
    model_name: str,
    continuous: list[str],
    binary: list[str],
    categorical: list[str],
) -> Pipeline:
    if model_name == "Logistic Regression":
        return Pipeline(
            [
                (
                    "preprocess",
                    make_preprocessor_custom(
                        continuous,
                        binary,
                        categorical,
                        scale=True,
                        add_missing_indicators=True,
                    ),
                ),
                ("model", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=RANDOM_STATE)),
            ]
        )
    if model_name == "XGBoost":
        return Pipeline(
            [
                (
                    "preprocess",
                    make_preprocessor_custom(
                        continuous,
                        binary,
                        categorical,
                        scale=False,
                        add_missing_indicators=True,
                    ),
                ),
                (
                    "model",
                    xgb.XGBClassifier(
                        n_estimators=500,
                        learning_rate=0.03,
                        max_depth=4,
                        min_child_weight=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    raise ValueError(model_name)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "auroc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "auprc": average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "brier_score": brier_score_loss(y, p),
        "observed_risk": float(y.mean()),
        "mean_predicted_risk": float(p.mean()),
        "observed_expected_ratio": float(y.mean() / p.mean()) if p.mean() > 0 else np.nan,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def simplified_predictors(landmark: int) -> list[str]:
    table = pd.read_csv(V10_DIR / "audit_v10_simplified_predictor_set.csv")
    row = table.loc[table["landmark_hours"].eq(landmark)].iloc[0]
    return [x.strip() for x in str(row["predictors"]).split(";") if x.strip()]


def conservative_uo_sensitivity() -> pd.DataFrame:
    data = pd.read_csv(V6_FILE, low_memory=False)
    rows = []
    thresholds = [0.50, 0.75, 0.90, 1.00]
    for threshold in thresholds:
        uo_evaluable = bool_mask(data["uo_evaluable"])
        coverage = pd.to_numeric(data["uo_observed_hour_fraction_7d"], errors="coerce")
        included = uo_evaluable & coverage.ge(threshold)
        sc = data.loc[included].copy()
        if sc.empty:
            continue
        scr = bool_mask(sc["aki_final"])
        uo = bool_mask(sc["uo_aki"])
        combined = scr | uo
        rows.append(
            {
                "uo_coverage_threshold": threshold,
                "n": int(len(sc)),
                "scr_only_aki_n": int(scr.sum()),
                "scr_only_aki_rate": float(scr.mean()),
                "uo_aki_n": int(uo.sum()),
                "uo_aki_rate": float(uo.mean()),
                "scr_or_uo_aki_n": int(combined.sum()),
                "scr_or_uo_aki_rate": float(combined.mean()),
                "uo_only_aki_n": int((uo & ~scr).sum()),
                "uo_only_aki_rate": float((uo & ~scr).mean()),
                "scr_and_uo_aki_n": int((uo & scr).sum()),
                "median_coverage": float(coverage.loc[included].median()),
                "median_uo_total_ml_7d": float(pd.to_numeric(sc["uo_total_ml_7d"], errors="coerce").median()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "audit_v14_conservative_uo_aki_sensitivity.csv", index=False)
    return out


def load_dynamic_for_rolling(landmark: int) -> pd.DataFrame:
    dyn = pd.read_csv(DYNAMIC_DIR / f"dataset_v4_{landmark}h.csv", low_memory=False)
    model = load_data(landmark)
    dyn["intime"] = pd.to_datetime(dyn["intime"], errors="coerce")
    dyn["icu_year"] = dyn["intime"].dt.year
    cols = ["subject_id", "hadm_id", "stay_id", "icu_year"]
    merged = model.merge(dyn[cols], on=["subject_id", "hadm_id", "stay_id"], how="left", validate="one_to_one")
    if merged["icu_year"].isna().any():
        raise ValueError(f"Missing ICU year for landmark {landmark}")
    return merged


def rolling_blocks(years: pd.Series) -> list[tuple[int, int]]:
    unique = np.sort(years.dropna().astype(int).unique())
    blocks = np.array_split(unique, ROLLING_BLOCKS)
    return [(int(block.min()), int(block.max())) for block in blocks if len(block)]


def rolling_temporal_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_tables = []
    for landmark in LANDMARKS:
        data = load_dynamic_for_rolling(landmark)
        predictors = [p for p in simplified_predictors(landmark) if p in data.columns]
        blocks = rolling_blocks(data["icu_year"])
        for i, (start, end) in enumerate(blocks):
            if i == 0:
                continue
            train = data.loc[data["icu_year"].lt(start)].copy()
            test = data.loc[data["icu_year"].between(start, end, inclusive="both")].copy()
            if len(train) < MIN_ROLLING_TRAIN_N or len(test) < MIN_ROLLING_TEST_N:
                continue
            if train[OUTCOME].nunique() < 2 or test[OUTCOME].nunique() < 2:
                continue
            model_name = SELECTED_MODELS[landmark]
            continuous, binary, categorical = identify_types_for_columns(train[predictors], predictors)
            pipe = model_pipeline(model_name, continuous, binary, categorical)
            pipe.fit(train[predictors], train[OUTCOME].astype(int).to_numpy())
            p = pipe.predict_proba(test[predictors])[:, 1]
            y = test[OUTCOME].astype(int).to_numpy()
            row = {
                "landmark_hours": landmark,
                "selected_model": model_name,
                "validation_block": i + 1,
                "train_year_max_lt": start,
                "validation_year_start": start,
                "validation_year_end": end,
                "train_n": len(train),
                "test_n": len(test),
                "train_event_rate": float(train[OUTCOME].mean()),
                "test_event_rate": float(test[OUTCOME].mean()),
                "predictor_n": len(predictors),
            }
            row.update(metrics(y, p))
            rows.append(row)
            pred = test[["subject_id", "hadm_id", "stay_id", "landmark_hours", "icu_year"]].copy()
            pred["y_true"] = y
            pred["prob_selected_parsimonious"] = p
            pred["validation_block"] = i + 1
            pred_tables.append(pred)
    perf = pd.DataFrame(rows)
    preds = pd.concat(pred_tables, ignore_index=True) if pred_tables else pd.DataFrame()
    perf.to_csv(OUTPUT_DIR / "model_v14_rolling_temporal_validation_performance.csv", index=False)
    preds.to_csv(OUTPUT_DIR / "model_v14_rolling_temporal_validation_predictions.csv", index=False)
    return perf, preds


def selected_parsimonious_finalization() -> pd.DataFrame:
    v10 = pd.read_csv(V10_DIR / "model_v10_full_vs_simplified_performance.csv")
    rows = []
    for landmark in LANDMARKS:
        selected_model = SELECTED_MODELS[landmark]
        row = v10.loc[
            v10["landmark_hours"].eq(landmark)
            & v10["variant"].eq("simplified")
            & v10["model"].eq(selected_model)
        ].iloc[0]
        predictors = simplified_predictors(landmark)
        rows.append(
            {
                "landmark_hours": landmark,
                "final_model_family": selected_model,
                "predictor_n": len(predictors),
                "predictors": "; ".join(predictors),
                "heldout_test_n": int(row["test_n"]),
                "heldout_test_event_rate": float(row["test_event_rate"]),
                "auroc": float(row["auroc"]),
                "auroc_ci_lower": float(row["auroc_ci_lower"]),
                "auroc_ci_upper": float(row["auroc_ci_upper"]),
                "auprc": float(row["auprc"]),
                "auprc_ci_lower": float(row["auprc_ci_lower"]),
                "auprc_ci_upper": float(row["auprc_ci_upper"]),
                "brier_score": float(row["brier_score"]),
                "brier_score_ci_lower": float(row["brier_score_ci_lower"]),
                "brier_score_ci_upper": float(row["brier_score_ci_upper"]),
                "finalization_note": "Selected parsimonious model retained for clinical translation; full models remain primary development benchmark.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "model_v14_selected_parsimonious_model_finalization.csv", index=False)
    return out


def association_specs() -> dict[int, list[dict[str, object]]]:
    return {
        0: [
            {"name": "preindex_lactate_ge_2", "source": "lab_pre24h_lactate_last", "type": "binary_ge", "threshold": 2.0, "label": "Pre-index lactate >=2 mmol/L"},
            {"name": "preindex_bun_per_10", "source": "lab_pre24h_bun_last", "type": "continuous_per", "scale": 10.0, "label": "Pre-index BUN, per 10 mg/dL"},
            {"name": "preindex_hemoglobin_per_minus_1", "source": "lab_pre24h_hemoglobin_last", "type": "continuous_negative", "scale": 1.0, "label": "Lower pre-index hemoglobin, per 1 g/dL"},
            {"name": "preindex_potassium_abnormal", "source": "lab_pre24h_potassium_last", "type": "outside", "low": 3.5, "high": 5.0, "label": "Pre-index potassium outside 3.5-5.0 mmol/L"},
        ],
        6: [
            {"name": "early_map_min_lt_65", "source": "vital_0_6h_map_min", "type": "binary_lt", "threshold": 65.0, "label": "0-6h minimum MAP <65 mmHg"},
            {"name": "early_lactate_max_ge_2", "source": "lab_0_6h_lactate_max", "type": "binary_ge", "threshold": 2.0, "label": "0-6h maximum lactate >=2 mmol/L"},
            {"name": "early_bun_max_per_10", "source": "lab_0_6h_bun_max", "type": "continuous_per", "scale": 10.0, "label": "0-6h maximum BUN, per 10 mg/dL"},
            {"name": "early_hemoglobin_min_per_minus_1", "source": "lab_0_6h_hemoglobin_min", "type": "continuous_negative", "scale": 1.0, "label": "Lower 0-6h hemoglobin, per 1 g/dL"},
            {"name": "early_potassium_abnormal", "source": "lab_0_6h_potassium_max", "alt_source": "lab_0_6h_potassium_min", "type": "potassium_window", "label": "0-6h potassium outside 3.5-5.0 mmol/L"},
        ],
        24: [
            {"name": "early_map_min_lt_65", "source": "vital_0_24h_map_min", "type": "binary_lt", "threshold": 65.0, "label": "0-24h minimum MAP <65 mmHg"},
            {"name": "early_lactate_max_ge_2", "source": "lab_0_24h_lactate_max", "type": "binary_ge", "threshold": 2.0, "label": "0-24h maximum lactate >=2 mmol/L"},
            {"name": "early_bun_max_per_10", "source": "lab_0_24h_bun_max", "type": "continuous_per", "scale": 10.0, "label": "0-24h maximum BUN, per 10 mg/dL"},
            {"name": "early_hemoglobin_min_per_minus_1", "source": "lab_0_24h_hemoglobin_min", "type": "continuous_negative", "scale": 1.0, "label": "Lower 0-24h hemoglobin, per 1 g/dL"},
            {"name": "early_potassium_abnormal", "source": "lab_0_24h_potassium_max", "alt_source": "lab_0_24h_potassium_min", "type": "potassium_window", "label": "0-24h potassium outside 3.5-5.0 mmol/L"},
        ],
    }


def build_marker(data: pd.DataFrame, spec: dict[str, object]) -> pd.Series:
    source = str(spec["source"])
    values = pd.to_numeric(data[source], errors="coerce")
    kind = spec["type"]
    if kind == "binary_ge":
        return values.ge(float(spec["threshold"])).astype(float).where(values.notna())
    if kind == "binary_lt":
        return values.lt(float(spec["threshold"])).astype(float).where(values.notna())
    if kind == "continuous_per":
        return values / float(spec["scale"])
    if kind == "continuous_negative":
        return -values / float(spec["scale"])
    if kind == "outside":
        return ((values.lt(float(spec["low"]))) | (values.gt(float(spec["high"])))).astype(float).where(values.notna())
    if kind == "potassium_window":
        high = pd.to_numeric(data[source], errors="coerce")
        low = pd.to_numeric(data[str(spec["alt_source"])], errors="coerce")
        observed = high.notna() | low.notna()
        abnormal = high.gt(5.0) | low.lt(3.5)
        return abnormal.astype(float).where(observed)
    raise ValueError(str(kind))


def adjusted_association_model(landmark: int, marker_name: str, label: str, marker: pd.Series, data: pd.DataFrame) -> dict[str, object]:
    covariates = [
        "anchor_age",
        "gender",
        "cardiac_surgery",
        "ckd",
        "dm",
        "charlson_score",
        "baseline_scr_at_landmark",
    ]
    available = [c for c in covariates if c in data.columns]
    work = data[[OUTCOME, *available]].copy()
    work[marker_name] = marker
    work = work.dropna(subset=[OUTCOME, marker_name])
    for c in available:
        if c in CATEGORICAL_COLUMNS or c == "gender":
            work[c] = work[c].astype("string").fillna("Unknown")
        else:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    if len(work) < 200 or work[OUTCOME].nunique() < 2 or work[marker_name].nunique(dropna=True) < 2:
        return {
            "landmark_hours": landmark,
            "marker": marker_name,
            "marker_label": label,
            "n": len(work),
            "event_n": int(work[OUTCOME].sum()) if len(work) else 0,
            "adjusted_odds_ratio": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "note": "not estimated: insufficient data or variation",
        }
    try:
        encoded = pd.get_dummies(work[[marker_name, *available]], drop_first=True, dummy_na=False)
        encoded = encoded.apply(pd.to_numeric, errors="coerce")
        encoded = encoded.fillna(encoded.median(numeric_only=True))
        y = work[OUTCOME].astype(int).to_numpy()
        x = encoded.to_numpy(dtype=float)
        columns = list(encoded.columns)
        marker_idx = columns.index(marker_name)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)
        model.fit(x, y)
        coef = float(model.coef_[0, marker_idx])

        rng = np.random.default_rng(RANDOM_STATE + landmark * 1000 + len(marker_name))
        boot_coefs = []
        attempts = 0
        while len(boot_coefs) < 300 and attempts < 900:
            attempts += 1
            idx = rng.integers(0, len(y), size=len(y))
            if len(np.unique(y[idx])) < 2 or len(np.unique(x[idx, marker_idx])) < 2:
                continue
            bmodel = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            bmodel.fit(x[idx], y[idx])
            boot_coefs.append(float(bmodel.coef_[0, marker_idx]))
        if boot_coefs:
            ci_lower = float(np.exp(np.quantile(boot_coefs, 0.025)))
            ci_upper = float(np.exp(np.quantile(boot_coefs, 0.975)))
            # Two-sided bootstrap sign test p value around a null log-OR of 0.
            signs = np.array(boot_coefs)
            p_value = float(2 * min(np.mean(signs <= 0), np.mean(signs >= 0)))
            p_value = min(max(p_value, 0.0), 1.0)
            if p_value == 0.0:
                p_value = 1.0 / (len(boot_coefs) + 1)
        else:
            ci_lower = np.nan
            ci_upper = np.nan
            p_value = np.nan
        return {
            "landmark_hours": landmark,
            "marker": marker_name,
            "marker_label": label,
            "n": int(len(work)),
            "event_n": int(y.sum()),
            "event_rate": float(y.mean()),
            "observed_marker_available_percent": float(marker.notna().mean() * 100),
            "marker_prevalence_or_median": float(work[marker_name].mean()) if set(work[marker_name].dropna().unique()).issubset({0.0, 1.0}) else float(work[marker_name].median()),
            "adjusted_odds_ratio": float(np.exp(coef)),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "note": "Adjusted logistic association using sklearn with bootstrap 95% CI; adjusted for age, sex, cardiac surgery, CKD, diabetes, Charlson score, and baseline creatinine when available; association only, not causal.",
        }
    except Exception as exc:
        return {
            "landmark_hours": landmark,
            "marker": marker_name,
            "marker_label": label,
            "n": len(work),
            "event_n": int(work[OUTCOME].sum()),
            "adjusted_odds_ratio": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "note": f"not estimated: {exc}",
        }


def secondary_association_analysis() -> pd.DataFrame:
    rows = []
    for landmark in LANDMARKS:
        data = load_data(landmark)
        for spec in association_specs()[landmark]:
            if str(spec["source"]) not in data.columns:
                continue
            if spec["type"] == "potassium_window" and str(spec["alt_source"]) not in data.columns:
                continue
            marker = build_marker(data, spec)
            rows.append(adjusted_association_model(landmark, str(spec["name"]), str(spec["label"]), marker, data))
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "analysis_v14_secondary_actionable_marker_associations.csv", index=False)
    return out


def make_figures(uo: pd.DataFrame, rolling: pd.DataFrame, final_models: pd.DataFrame, assoc: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    ax.plot(uo["uo_coverage_threshold"], uo["scr_only_aki_rate"] * 100, marker="o", label="SCr-only")
    ax.plot(uo["uo_coverage_threshold"], uo["uo_aki_rate"] * 100, marker="o", label="UO-only criterion")
    ax.plot(uo["uo_coverage_threshold"], uo["scr_or_uo_aki_rate"] * 100, marker="o", label="SCr-or-UO")
    ax.set_xlabel("Minimum urine-output hourly coverage fraction")
    ax.set_ylabel("AKI incidence (%)")
    ax.set_title("Conservative urine-output AKI sensitivity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_v14_conservative_uo_aki_sensitivity.png", dpi=300)
    plt.close(fig)

    if not rolling.empty:
        fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=180)
        for landmark, group in rolling.groupby("landmark_hours"):
            group = group.sort_values("validation_year_start")
            ax.plot(
                group["validation_year_start"].astype(str) + "-" + group["validation_year_end"].astype(str),
                group["auroc"],
                marker="o",
                label=f"{landmark} h",
            )
        ax.set_ylabel("AUROC")
        ax.set_xlabel("Validation year block")
        ax.set_title("Rolling temporal validation of selected parsimonious models")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "figure_v14_rolling_temporal_auroc.png", dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)
    x = np.arange(len(final_models))
    ax.bar(x, final_models["auroc"], color="#4c78a8")
    ax.errorbar(
        x,
        final_models["auroc"],
        yerr=[
            final_models["auroc"] - final_models["auroc_ci_lower"],
            final_models["auroc_ci_upper"] - final_models["auroc"],
        ],
        fmt="none",
        color="black",
        capsize=3,
    )
    ax.set_xticks(x, labels=[f"{int(v)} h\n{m}" for v, m in zip(final_models["landmark_hours"], final_models["final_model_family"])])
    ax.set_ylabel("Held-out AUROC")
    ax.set_title("Final selected parsimonious models")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_v14_selected_parsimonious_model_performance.png", dpi=300)
    plt.close(fig)

    plot_assoc = assoc.dropna(subset=["adjusted_odds_ratio"]).copy()
    if not plot_assoc.empty:
        plot_assoc["label"] = plot_assoc["landmark_hours"].astype(str) + "h: " + plot_assoc["marker_label"]
        plot_assoc = plot_assoc.sort_values(["landmark_hours", "adjusted_odds_ratio"])
        fig, ax = plt.subplots(figsize=(8.8, max(5.0, len(plot_assoc) * 0.33)), dpi=180)
        y = np.arange(len(plot_assoc))
        ax.errorbar(
            plot_assoc["adjusted_odds_ratio"],
            y,
            xerr=[
                plot_assoc["adjusted_odds_ratio"] - plot_assoc["ci_lower"],
                plot_assoc["ci_upper"] - plot_assoc["adjusted_odds_ratio"],
            ],
            fmt="o",
            color="#e45756",
            ecolor="#f4a3a3",
            capsize=2,
        )
        ax.axvline(1.0, color="black", linestyle="--", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_yticks(y, labels=plot_assoc["label"])
        ax.set_xlabel("Adjusted odds ratio, log scale")
        ax.set_title("Secondary association analysis of actionable markers")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "figure_v14_secondary_actionable_marker_associations.png", dpi=300)
        plt.close(fig)


def write_readme(uo: pd.DataFrame, rolling: pd.DataFrame, final_models: pd.DataFrame, assoc: pd.DataFrame) -> None:
    lines = ["# v14 final sensitivity and actionable-marker analysis", ""]
    lines.append("## Conservative urine-output AKI sensitivity")
    lines.append("")
    for row in uo.itertuples():
        lines.append(
            f"- Coverage >= {row.uo_coverage_threshold:.0%}: n={row.n:,}; SCr-only AKI {row.scr_only_aki_rate*100:.1f}%; "
            f"UO AKI {row.uo_aki_rate*100:.1f}%; SCr-or-UO AKI {row.scr_or_uo_aki_rate*100:.1f}%; "
            f"UO-only AKI {row.uo_only_aki_rate*100:.1f}%."
        )
    lines.append("")
    lines.append("## Rolling temporal validation")
    lines.append("")
    if rolling.empty:
        lines.append("No rolling temporal blocks met minimum sample-size criteria.")
    else:
        lines.append("| Landmark | Validation years | Train n | Test n | AUROC | AUPRC | Brier | Calibration slope |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        for row in rolling.sort_values(["landmark_hours", "validation_year_start"]).itertuples():
            lines.append(
                f"| {int(row.landmark_hours)} h | {int(row.validation_year_start)}-{int(row.validation_year_end)} | "
                f"{int(row.train_n):,} | {int(row.test_n):,} | {row.auroc:.3f} | {row.auprc:.3f} | "
                f"{row.brier_score:.3f} | {row.calibration_slope:.2f} |"
            )
    lines.append("")
    lines.append("## Selected parsimonious model finalization")
    lines.append("")
    for row in final_models.itertuples():
        lines.append(
            f"- {int(row.landmark_hours)} h: {row.final_model_family}, {int(row.predictor_n)} predictors; "
            f"held-out AUROC {row.auroc:.3f} ({row.auroc_ci_lower:.3f}-{row.auroc_ci_upper:.3f}), "
            f"AUPRC {row.auprc:.3f}, Brier {row.brier_score:.3f}."
        )
    lines.append("")
    lines.append("## Secondary association analysis: cautious wording")
    lines.append("")
    lines.append("These estimates are adjusted prognostic associations. They should be described as potentially actionable markers, not causal or proven modifiable effects.")
    lines.append("")
    lines.append("| Landmark | Marker | Adjusted OR | 95% CI | P value |")
    lines.append("|---:|---|---:|---:|---:|")
    for row in assoc.dropna(subset=["adjusted_odds_ratio"]).itertuples():
        lines.append(
            f"| {int(row.landmark_hours)} h | {row.marker_label} | {row.adjusted_odds_ratio:.2f} | "
            f"{row.ci_lower:.2f}-{row.ci_upper:.2f} | {row.p_value:.3g} |"
        )
    lines.append("")
    lines.append("Suggested manuscript wording: Early postoperative physiological derangements, including hypotension, hyperlactatemia, azotemia, anemia, and potassium abnormalities, were associated with subsequent AKI after adjustment for baseline risk factors. These findings identify clinically actionable risk markers but should not be interpreted as causal intervention effects.")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    for name in [
        "audit_v14_conservative_uo_aki_sensitivity.csv",
        "model_v14_rolling_temporal_validation_performance.csv",
        "model_v14_rolling_temporal_validation_predictions.csv",
        "model_v14_selected_parsimonious_model_finalization.csv",
        "analysis_v14_secondary_actionable_marker_associations.csv",
        "figure_v14_conservative_uo_aki_sensitivity.png",
        "figure_v14_rolling_temporal_auroc.png",
        "figure_v14_selected_parsimonious_model_performance.png",
        "figure_v14_secondary_actionable_marker_associations.png",
    ]:
        lines.append(f"- `{name}`")
    (OUTPUT_DIR / "audit_v14_results_brief.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    uo = conservative_uo_sensitivity()
    rolling, _ = rolling_temporal_validation()
    final_models = selected_parsimonious_finalization()
    assoc = secondary_association_analysis()
    make_figures(uo, rolling, final_models, assoc)
    write_readme(uo, rolling, final_models, assoc)
    print(f"v14 final sensitivity analyses complete: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
