"""v5.4 small pre-submission optimizations.

Adds:
1) subject-level temporal validation sensitivity analysis;
2) selected-model cardiac vs non-cardiac subgroup performance table;
3) clinically interpretable threshold / alert-burden table;
4) manuscript-ready note strengthening the 24 h no-creatinine interpretation.

This script does not overwrite v5.1-v5.3 outputs.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from develop_models_v5 import (  # noqa: E402
    OUTCOME,
    RANDOM_STATE,
    calibration_intercept_slope,
    identify_types,
    load_data,
)
from extend_models_v5_1 import (  # noqa: E402
    MODEL_STYLES,
    bootstrap_ci,
    model_definitions,
)


PROJECT_ROOT = SCRIPT_DIR.parent
DYNAMIC_DIR = PROJECT_ROOT / "outputs" / "dynamic_v4"
V5_1_DIR = PROJECT_ROOT / "outputs" / "modeling_v5_1"
V5_2_DIR = PROJECT_ROOT / "outputs" / "modeling_v5_2_no_creatinine"
V5_3_DIR = PROJECT_ROOT / "outputs" / "modeling_v5_3_preindex_baseline"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "modeling_v5_4_small_optimizations"
LANDMARKS = [0, 6, 24]
SELECTED_MODELS = {0: "XGBoost", 6: "XGBoost", 24: "Logistic Regression"}
SELECTED_PROB_COLUMNS = {
    "Logistic Regression": "prob_logistic_regression",
    "Random Forest": "prob_random_forest",
    "XGBoost": "prob_xgboost",
    "LightGBM": "prob_lightgbm",
}
CLINICAL_THRESHOLDS = [0.20, 0.30, 0.50]
TEMPORAL_BOOTSTRAPS = 500

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


def slug(model: str) -> str:
    return model.lower().replace(" ", "_")


def load_subject_first_year() -> pd.DataFrame:
    dyn0 = pd.read_csv(DYNAMIC_DIR / "dataset_v4_0h.csv", usecols=["subject_id", "hadm_id", "stay_id", "intime"], low_memory=False)
    dyn0["intime"] = pd.to_datetime(dyn0["intime"], errors="coerce")
    if dyn0["intime"].isna().any():
        raise ValueError("Missing or invalid intime values in dataset_v4_0h.csv")
    subject_year = (
        dyn0.sort_values(["subject_id", "intime"])
        .groupby("subject_id", as_index=False)["intime"]
        .first()
    )
    subject_year["first_icu_year"] = subject_year["intime"].dt.year.astype(int)
    return subject_year[["subject_id", "first_icu_year"]]


def choose_temporal_cutoff(subject_year: pd.DataFrame, target_test_fraction: float = 0.20) -> tuple[int, pd.DataFrame]:
    years = sorted(subject_year["first_icu_year"].unique())
    rows = []
    best = None
    n_subjects = subject_year["subject_id"].nunique()
    for cutoff in years[1:]:
        train = subject_year["first_icu_year"].lt(cutoff)
        test = subject_year["first_icu_year"].ge(cutoff)
        train_n = int(train.sum())
        test_n = int(test.sum())
        if train_n == 0 or test_n == 0:
            continue
        test_fraction = test_n / n_subjects
        row = {
            "candidate_cutoff_year": cutoff,
            "train_subject_n": train_n,
            "test_subject_n": test_n,
            "test_subject_fraction": test_fraction,
            "absolute_deviation_from_20_percent": abs(test_fraction - target_test_fraction),
        }
        rows.append(row)
        score = row["absolute_deviation_from_20_percent"]
        if best is None or score < best[0]:
            best = (score, cutoff)
    if best is None:
        raise ValueError("Could not choose a temporal cutoff")
    return int(best[1]), pd.DataFrame(rows)


def youden_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    finite = np.isfinite(thresholds)
    j = tpr - fpr
    eligible = np.where(finite)[0]
    best = eligible[np.argmax(j[eligible])]
    return float(thresholds[best])


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = len(y_true)
    return {
        "threshold": float(threshold),
        "n": int(n),
        "event_n": int(y_true.sum()),
        "event_rate": float(y_true.mean()),
        "alert_n": int(pred.sum()),
        "alert_rate": float(pred.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "precision_ppv": precision_score(y_true, pred, zero_division=0),
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier_score": brier_score_loss(y, p),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def run_temporal_validation() -> None:
    subject_year = load_subject_first_year()
    cutoff, cutoff_audit = choose_temporal_cutoff(subject_year)
    cutoff_audit.to_csv(OUTPUT_DIR / "model_v5_4_temporal_cutoff_candidates.csv", index=False)

    performance_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for landmark in LANDMARKS:
        data = load_data(landmark)
        data = data.merge(subject_year, on="subject_id", how="left", validate="many_to_one")
        if data["first_icu_year"].isna().any():
            raise ValueError(f"Missing first_icu_year after merge at {landmark} h")
        train_mask = data["first_icu_year"].lt(cutoff)
        test_mask = data["first_icu_year"].ge(cutoff)
        train = data.loc[train_mask].copy()
        test = data.loc[test_mask].copy()
        if set(train["subject_id"]).intersection(set(test["subject_id"])):
            raise AssertionError("Temporal split leaked subject IDs")

        split_rows.append({
            "landmark_hours": landmark,
            "temporal_cutoff_year": cutoff,
            "rule": "train subject first ICU year < cutoff; test subject first ICU year >= cutoff",
            "train_n": len(train),
            "test_n": len(test),
            "train_subject_n": train["subject_id"].nunique(),
            "test_subject_n": test["subject_id"].nunique(),
            "train_event_n": int(train[OUTCOME].sum()),
            "test_event_n": int(test[OUTCOME].sum()),
            "train_event_rate": float(train[OUTCOME].mean()),
            "test_event_rate": float(test[OUTCOME].mean()),
            "train_year_min": int(train["first_icu_year"].min()),
            "train_year_max": int(train["first_icu_year"].max()),
            "test_year_min": int(test["first_icu_year"].min()),
            "test_year_max": int(test["first_icu_year"].max()),
        })

        continuous, binary, categorical = identify_types(data.drop(columns=["first_icu_year"]))
        models = model_definitions(continuous, binary, categorical)
        x_train = train.drop(columns=[OUTCOME, "first_icu_year"])
        y_train = train[OUTCOME].to_numpy(dtype=int)
        x_test = test.drop(columns=[OUTCOME, "first_icu_year"])
        y_test = test[OUTCOME].to_numpy(dtype=int)
        subjects_test = test["subject_id"].to_numpy(dtype=int)

        for model_name, pipe in models.items():
            pipe.fit(x_train, y_train)
            p_train = pipe.predict_proba(x_train)[:, 1]
            p_test = pipe.predict_proba(x_test)[:, 1]
            threshold = youden_threshold(y_train, p_train)
            row: dict[str, object] = {
                "landmark_hours": landmark,
                "model": model_name,
                "selected_model_for_landmark": model_name == SELECTED_MODELS[landmark],
                "temporal_cutoff_year": cutoff,
                "train_n": len(train),
                "test_n": len(test),
                "train_subject_n": train["subject_id"].nunique(),
                "test_subject_n": test["subject_id"].nunique(),
                "train_event_rate": float(y_train.mean()),
                "test_event_rate": float(y_test.mean()),
                "youden_threshold_training": threshold,
                **point_metrics(y_test, p_test),
                **{f"youden_{k}": v for k, v in threshold_metrics(y_test, p_test, threshold).items()},
            }
            ci = bootstrap_ci(
                y_test,
                p_test,
                subjects_test,
                TEMPORAL_BOOTSTRAPS,
                seed=RANDOM_STATE + 54000 + landmark * 100 + len(performance_rows),
            )
            row.update(ci)
            performance_rows.append(row)

    pd.DataFrame(performance_rows).to_csv(OUTPUT_DIR / "model_v5_4_temporal_validation_performance.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUTPUT_DIR / "model_v5_4_temporal_split_audit.csv", index=False)


def build_cardiac_noncardiac_table() -> None:
    subgroup = pd.read_csv(V5_1_DIR / "model_v5_1_subgroup_performance.csv")
    rows = []
    wanted = [
        ("cardiac_surgery", "yes", "Cardiac surgery"),
        ("cardiac_surgery", "no", "Non-cardiac surgery"),
    ]
    for landmark, selected_model in SELECTED_MODELS.items():
        for dimension, level, label in wanted:
            match = subgroup[
                (subgroup["landmark_hours"].eq(landmark))
                & (subgroup["model"].eq(selected_model))
                & (subgroup["subgroup_dimension"].eq(dimension))
                & (subgroup["subgroup_level"].astype(str).eq(level))
            ]
            if match.empty:
                continue
            row = match.iloc[0].to_dict()
            row["selected_model"] = selected_model
            row["clinical_group"] = label
            rows.append(row)
    cols = [
        "landmark_hours", "selected_model", "clinical_group", "n", "subject_n",
        "event_n", "event_rate", "auroc", "auroc_ci_lower", "auroc_ci_upper",
        "auprc", "auprc_ci_lower", "auprc_ci_upper", "brier_score",
        "sensitivity_0_5", "specificity_0_5",
    ]
    out = pd.DataFrame(rows)
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(OUTPUT_DIR / "model_v5_4_cardiac_noncardiac_selected_performance.csv", index=False)


def build_threshold_alert_burden_table() -> None:
    rows = []
    for landmark, model_name in SELECTED_MODELS.items():
        preds = pd.read_csv(V5_1_DIR / f"model_v5_1_{landmark}h_test_predictions.csv")
        y = preds["y_true"].to_numpy(dtype=int)
        p = preds[SELECTED_PROB_COLUMNS[model_name]].to_numpy(dtype=float)
        thresholds = [(f"clinical_{t:.2f}", t) for t in CLINICAL_THRESHOLDS]
        youden_col = f"youden_threshold_{slug(model_name)}"
        if youden_col not in preds.columns:
            # Columns use lower snake case for logistic regression and model names.
            youden_col = [c for c in preds.columns if c.startswith("youden_threshold_") and slug(model_name).replace("_", "") in c.replace("_", "")]
            if isinstance(youden_col, list):
                youden = float(preds[youden_col[0]].iloc[0])
            else:
                youden = youden_threshold(y, p)
        else:
            youden = float(preds[youden_col].iloc[0])
        thresholds.append(("development_youden", youden))
        for threshold_label, threshold in thresholds:
            row = {
                "landmark_hours": landmark,
                "selected_model": model_name,
                "threshold_label": threshold_label,
                **threshold_metrics(y, p, threshold),
            }
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "model_v5_4_threshold_alert_burden.csv", index=False)


def build_no_creatinine_discussion_note() -> None:
    no_creat = pd.read_csv(V5_2_DIR / "model_v5_2_full_vs_no_creatinine_paired_comparison.csv")
    preindex = pd.read_csv(V5_3_DIR / "model_v5_3_full_vs_preindex_paired_comparison.csv")
    lr_nc = no_creat[no_creat["model"].eq("Logistic Regression") & no_creat["landmark_hours"].eq(24)].iloc[0]
    lr_pi = preindex[preindex["model"].eq("Logistic Regression") & preindex["landmark_hours"].eq(24)].iloc[0]
    lines = [
        "# v5.4 no-creatinine interpretation note",
        "",
        "Purpose: strengthen manuscript interpretation without changing the scientific results.",
        "",
        "Suggested Discussion language:",
        "",
        (
            "Creatinine-related variables were intentionally examined because the 24 h prediction task is close in time "
            "to a creatinine-defined AKI outcome. In the 24 h sensitivity analysis, removing all creatinine-derived, "
            f"baseline SCr, and baseline-to-ICU timing predictors reduced logistic-regression AUROC from {lr_nc['full_auroc']:.3f} "
            f"to {lr_nc['no_creatinine_auroc']:.3f} (paired difference, {lr_nc['delta_auroc']:+.3f}; "
            f"95% CI, {lr_nc['delta_auroc_ci_lower']:+.3f} to {lr_nc['delta_auroc_ci_upper']:+.3f}). "
            "This attenuation supports the interpretation that part of the 24 h model performance reflects early kidney-function "
            "trajectory and temporal proximity to the serum-creatinine outcome definition. However, no-creatinine models retained "
            "moderate discrimination, suggesting that demographic, comorbidity, surgical, hematologic, and physiologic information "
            "also contributed nontrivial predictive signal."
        ),
        "",
        (
            "In contrast, restricting the 24 h analysis to patients whose baseline SCr was measured during the 7 days before ICU "
            f"admission did not materially change discrimination for logistic regression ({lr_pi['full_model_auroc']:.3f} vs "
            f"{lr_pi['preindex_model_auroc']:.3f}; paired difference, {lr_pi['delta_auroc']:+.3f}; "
            f"95% CI, {lr_pi['delta_auroc_ci_lower']:+.3f} to {lr_pi['delta_auroc_ci_upper']:+.3f}), "
            "supporting robustness to the baseline-creatinine source definition."
        ),
        "",
        "Recommended caution:",
        "",
        "- Keep SHAP statements as model attribution only, not causal or modifiable-factor evidence.",
        "- Do not describe the 24 h full model as clinically deployable without recalibration and external validation.",
    ]
    (OUTPUT_DIR / "audit_v5_4_no_creatinine_discussion_note.md").write_text("\n".join(lines), encoding="utf-8")


def build_readme() -> None:
    readme = """# v5.4 small pre-submission optimizations

Generated outputs:

- `model_v5_4_temporal_validation_performance.csv`: subject-level temporal validation sensitivity analysis. Subjects were assigned by first ICU year; earlier subjects were used for model development and later subjects for validation.
- `model_v5_4_temporal_split_audit.csv`: split sizes, event rates, year ranges, and no-overlap audit.
- `model_v5_4_temporal_cutoff_candidates.csv`: candidate temporal cutoffs considered when targeting an approximately 20% later-period test subject fraction.
- `model_v5_4_cardiac_noncardiac_selected_performance.csv`: selected-model performance in cardiac and non-cardiac groups.
- `model_v5_4_threshold_alert_burden.csv`: clinical threshold examples with alert rate, PPV, NPV, sensitivity, specificity, and confusion counts.
- `audit_v5_4_no_creatinine_discussion_note.md`: manuscript-ready language clarifying the 24 h no-creatinine sensitivity result.

No machine-learning manuscript conclusions were changed automatically.
"""
    (OUTPUT_DIR / "audit_v5_4_small_optimizations_readme.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_temporal_validation()
    build_cardiac_noncardiac_table()
    build_threshold_alert_burden_table()
    build_no_creatinine_discussion_note()
    build_readme()
    print(f"Wrote v5.4 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
