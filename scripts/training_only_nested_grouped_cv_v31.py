"""Training-only nested subject-grouped model-selection audit for severe SCr-AKI.

This audit deliberately leaves the locked 20% subject-level test partition
untouched.  Within the original 80% training partition it compares the two
prespecified model families used by v27 (logistic regression and XGBoost):

* outer 5-fold stratified subject-grouped CV estimates the performance of the
  complete model-family selection procedure;
* inner 4-fold stratified subject-grouped CV selects the family with the
  highest pooled out-of-fold AUROC in each outer training fold; and
* a separate 5-fold training-only OOF comparison supplies the family that
  would be chosen before opening the locked test set.

No hyperparameters are tuned, and preprocessing/model specifications are the
same as in the locked v27 severe-AKI analysis.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from final_sensitivity_and_actionable_analysis_v14 import (  # noqa: E402
    metrics,
    model_pipeline,
    simplified_predictors,
)
from recalibration_and_measurement_intensity_v13 import (  # noqa: E402
    identify_types_for_columns,
)


V27 = ROOT / "outputs" / "modeling_v27_severity_recovery"
OUT = ROOT / "outputs" / "modeling_v31_nested_grouped_cv"
SPLIT_FILE = V27 / "audit_v27_subject_split_assignment.csv"
OUTCOME = "outcome_severe_scr_after_landmark_to_7d"
LANDMARKS = [0, 6, 24]
MODELS = ["Logistic Regression", "XGBoost"]
REPORTED_MODEL = {0: "XGBoost", 6: "XGBoost", 24: "Logistic Regression"}
RANDOM_STATE = 20250711
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAPS = 1000


def predictor_set(data: pd.DataFrame, landmark: int) -> list[str]:
    locked = [p for p in simplified_predictors(landmark) if p in data.columns]
    if landmark == 0:
        return locked
    additions = [
        "prior_stage1_aki_by_landmark",
        "hours_since_first_aki_at_landmark",
        "current_scr_stage_at_landmark",
        "current_scr_at_landmark",
        "current_scr_ratio_at_landmark",
        "scr_measurement_n_by_landmark",
    ]
    return locked + [p for p in additions if p in data.columns and p not in locked]


def grouped_oof(
    data: pd.DataFrame,
    predictors: list[str],
    model_name: str,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = data[OUTCOME].astype(int).to_numpy()
    groups = data["subject_id"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    probability = np.full(len(data), np.nan, dtype=float)
    fold_id = np.full(len(data), -1, dtype=int)
    continuous, binary, categorical = identify_types_for_columns(data, predictors)
    for fold, (fit_idx, val_idx) in enumerate(
        splitter.split(data[predictors], y, groups), start=1
    ):
        fit_groups = set(groups[fit_idx])
        val_groups = set(groups[val_idx])
        if fit_groups & val_groups:
            raise AssertionError("Subject overlap inside grouped CV fold")
        pipeline = model_pipeline(model_name, continuous, binary, categorical)
        pipeline.fit(data.iloc[fit_idx][predictors], y[fit_idx])
        probability[val_idx] = pipeline.predict_proba(
            data.iloc[val_idx][predictors]
        )[:, 1]
        fold_id[val_idx] = fold
    if np.isnan(probability).any() or (fold_id < 1).any():
        raise AssertionError("Incomplete out-of-fold predictions")
    return probability, fold_id


def clustered_bootstrap_ci(
    frame: pd.DataFrame, probability_column: str, seed: int
) -> dict[str, tuple[float, float, int]]:
    frame = frame.reset_index(drop=True)
    grouped = {
        subject: group.index.to_numpy()
        for subject, group in frame.groupby("subject_id", sort=False)
    }
    subjects = np.asarray(list(grouped), dtype=object)
    y_all = frame[OUTCOME].astype(int).to_numpy()
    p_all = frame[probability_column].astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    values = {"auroc": [], "auprc": [], "brier_score": []}
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        idx = np.concatenate([grouped[s] for s in sampled])
        if len(np.unique(y_all[idx])) < 2:
            continue
        values["auroc"].append(float(roc_auc_score(y_all[idx], p_all[idx])))
        values["auprc"].append(float(average_precision_score(y_all[idx], p_all[idx])))
        values["brier_score"].append(float(brier_score_loss(y_all[idx], p_all[idx])))
    result: dict[str, tuple[float, float, int]] = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        result[key] = (
            float(np.quantile(arr, 0.025)),
            float(np.quantile(arr, 0.975)),
            int(len(arr)),
        )
    return result


def run_landmark(
    landmark: int, split_assignment: pd.DataFrame
) -> tuple[list[dict], list[dict], pd.DataFrame, list[dict], pd.DataFrame]:
    data = pd.read_csv(V27 / f"dataset_v27_severe_{landmark}h.csv", low_memory=False)
    split = split_assignment[["stay_id", "split"]]
    data = data.merge(split, on="stay_id", how="left", validate="one_to_one")
    if data["split"].isna().any():
        raise AssertionError("Risk-set row missing from locked v27 split assignment")
    train = data.loc[data["split"].eq("train")].reset_index(drop=True)
    test = data.loc[data["split"].eq("test")].reset_index(drop=True)
    predictors = predictor_set(train, landmark)
    y = train[OUTCOME].astype(int).to_numpy()
    groups = train["subject_id"].to_numpy()
    continuous, binary, categorical = identify_types_for_columns(train, predictors)

    outer = StratifiedGroupKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE + landmark,
    )
    inner_rows: list[dict] = []
    outer_rows: list[dict] = []
    nested_predictions: list[pd.DataFrame] = []

    for outer_fold, (outer_fit_idx, outer_val_idx) in enumerate(
        outer.split(train[predictors], y, groups), start=1
    ):
        outer_fit = train.iloc[outer_fit_idx].reset_index(drop=True)
        outer_val = train.iloc[outer_val_idx].reset_index(drop=True)
        if set(outer_fit["subject_id"]) & set(outer_val["subject_id"]):
            raise AssertionError("Subject overlap between outer fit and validation")

        candidate_scores: dict[str, dict[str, float]] = {}
        for model_name in MODELS:
            inner_prob, inner_fold = grouped_oof(
                outer_fit,
                predictors,
                model_name,
                n_splits=INNER_FOLDS,
                random_state=RANDOM_STATE + landmark * 100 + outer_fold,
            )
            score = metrics(outer_fit[OUTCOME].to_numpy(int), inner_prob)
            candidate_scores[model_name] = score
            inner_rows.append(
                {
                    "landmark_hours": landmark,
                    "outer_fold": outer_fold,
                    "candidate_model": model_name,
                    "inner_fold_n": INNER_FOLDS,
                    "inner_training_rows": len(outer_fit),
                    "inner_training_subjects": outer_fit["subject_id"].nunique(),
                    "inner_event_n": int(outer_fit[OUTCOME].sum()),
                    "inner_oof_auroc": score["auroc"],
                    "inner_oof_auprc": score["auprc"],
                    "inner_oof_brier": score["brier_score"],
                    "inner_oof_calibration_intercept": score["calibration_intercept"],
                    "inner_oof_calibration_slope": score["calibration_slope"],
                    "inner_group_overlap_n": 0,
                    "selected_in_outer_fold": False,
                }
            )

        # Model family is selected exclusively by pooled inner-OOF AUROC.
        selected = max(MODELS, key=lambda name: candidate_scores[name]["auroc"])
        for row in inner_rows[-len(MODELS) :]:
            row["selected_in_outer_fold"] = row["candidate_model"] == selected

        pipeline = model_pipeline(selected, continuous, binary, categorical)
        pipeline.fit(outer_fit[predictors], outer_fit[OUTCOME].astype(int))
        outer_prob = pipeline.predict_proba(outer_val[predictors])[:, 1]
        score = metrics(outer_val[OUTCOME].to_numpy(int), outer_prob)
        outer_rows.append(
            {
                "landmark_hours": landmark,
                "outer_fold": outer_fold,
                "selected_model": selected,
                "outer_validation_n": len(outer_val),
                "outer_validation_subjects": outer_val["subject_id"].nunique(),
                "outer_event_n": int(outer_val[OUTCOME].sum()),
                "outer_event_rate": float(outer_val[OUTCOME].mean()),
                "outer_auroc": score["auroc"],
                "outer_auprc": score["auprc"],
                "outer_brier": score["brier_score"],
                "outer_calibration_intercept": score["calibration_intercept"],
                "outer_calibration_slope": score["calibration_slope"],
                "outer_group_overlap_n": 0,
            }
        )
        pred = outer_val[["subject_id", "hadm_id", "stay_id", OUTCOME]].copy()
        pred["landmark_hours"] = landmark
        pred["outer_fold"] = outer_fold
        pred["selected_model"] = selected
        pred["nested_oof_probability"] = outer_prob
        nested_predictions.append(pred)

    nested = pd.concat(nested_predictions, ignore_index=True)
    if len(nested) != len(train) or nested["stay_id"].nunique() != len(train):
        raise AssertionError("Nested OOF predictions do not cover training rows once")

    # Training-only family comparison used for the pre-test recommendation.
    candidate_rows: list[dict] = []
    candidate_predictions = train[["subject_id", "hadm_id", "stay_id", OUTCOME]].copy()
    candidate_predictions["landmark_hours"] = landmark
    for model_name in MODELS:
        probability, fold_id = grouped_oof(
            train,
            predictors,
            model_name,
            n_splits=OUTER_FOLDS,
            random_state=RANDOM_STATE + 1000 + landmark,
        )
        safe = model_name.lower().replace(" ", "_")
        candidate_predictions[f"probability_{safe}"] = probability
        candidate_predictions[f"fold_{safe}"] = fold_id
        score = metrics(y, probability)
        candidate_rows.append(
            {
                "landmark_hours": landmark,
                "candidate_model": model_name,
                "training_rows": len(train),
                "training_subjects": train["subject_id"].nunique(),
                "training_event_n": int(train[OUTCOME].sum()),
                "training_event_rate": float(train[OUTCOME].mean()),
                "predictor_n": len(predictors),
                "cv_fold_n": OUTER_FOLDS,
                "training_oof_auroc": score["auroc"],
                "training_oof_auprc": score["auprc"],
                "training_oof_brier": score["brier_score"],
                "training_oof_calibration_intercept": score["calibration_intercept"],
                "training_oof_calibration_slope": score["calibration_slope"],
                "subject_group_overlap_n": 0,
                "locked_test_rows_used_for_selection": 0,
            }
        )

    recommended = max(
        candidate_rows, key=lambda row: row["training_oof_auroc"]
    )["candidate_model"]
    for row in candidate_rows:
        row["training_only_recommended_model"] = recommended
        row["recommended_by_training_oof_auroc"] = row["candidate_model"] == recommended
        row["previously_reported_model"] = REPORTED_MODEL[landmark]
        row["recommendation_concordant_with_reported"] = recommended == REPORTED_MODEL[landmark]
        row["locked_test_n_embargoed"] = len(test)
        row["locked_test_subjects_embargoed"] = test["subject_id"].nunique()

    return inner_rows, outer_rows, nested, candidate_rows, candidate_predictions


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(SPLIT_FILE)
    if split["stay_id"].duplicated().any():
        raise AssertionError("Duplicate stay_id in split assignment")
    subject_split_n = split.groupby("subject_id")["split"].nunique()
    if subject_split_n.max() != 1:
        raise AssertionError("A subject appears in both locked train and test partitions")

    all_inner: list[dict] = []
    all_outer: list[dict] = []
    all_nested: list[pd.DataFrame] = []
    all_candidates: list[dict] = []
    all_candidate_predictions: list[pd.DataFrame] = []
    for landmark in LANDMARKS:
        print(f"Running nested grouped CV at {landmark} h", flush=True)
        inner, outer, nested, candidates, candidate_predictions = run_landmark(
            landmark, split
        )
        all_inner.extend(inner)
        all_outer.extend(outer)
        all_nested.append(nested)
        all_candidates.extend(candidates)
        all_candidate_predictions.append(candidate_predictions)

    inner_df = pd.DataFrame(all_inner)
    outer_df = pd.DataFrame(all_outer)
    nested_df = pd.concat(all_nested, ignore_index=True)
    candidate_df = pd.DataFrame(all_candidates)
    candidate_pred_df = pd.concat(all_candidate_predictions, ignore_index=True)

    summary_rows: list[dict] = []
    for landmark in LANDMARKS:
        nested = nested_df.loc[nested_df["landmark_hours"].eq(landmark)].copy()
        score = metrics(
            nested[OUTCOME].to_numpy(int), nested["nested_oof_probability"].to_numpy(float)
        )
        ci = clustered_bootstrap_ci(
            nested, "nested_oof_probability", RANDOM_STATE + 2000 + landmark
        )
        frequency = (
            outer_df.loc[outer_df["landmark_hours"].eq(landmark), "selected_model"]
            .value_counts()
            .to_dict()
        )
        candidates = candidate_df.loc[candidate_df["landmark_hours"].eq(landmark)]
        recommended = candidates.loc[
            candidates["recommended_by_training_oof_auroc"].astype(bool),
            "candidate_model",
        ].iloc[0]
        lr_auc = float(
            candidates.loc[candidates["candidate_model"].eq("Logistic Regression"), "training_oof_auroc"].iloc[0]
        )
        xgb_auc = float(
            candidates.loc[candidates["candidate_model"].eq("XGBoost"), "training_oof_auroc"].iloc[0]
        )
        summary_rows.append(
            {
                "landmark_hours": landmark,
                "training_n": len(nested),
                "training_subjects": nested["subject_id"].nunique(),
                "training_event_n": int(nested[OUTCOME].sum()),
                "training_event_rate": float(nested[OUTCOME].mean()),
                "nested_outer_fold_n": OUTER_FOLDS,
                "nested_inner_fold_n": INNER_FOLDS,
                "nested_selection_oof_auroc": score["auroc"],
                "nested_selection_oof_auroc_ci95_low": ci["auroc"][0],
                "nested_selection_oof_auroc_ci95_high": ci["auroc"][1],
                "nested_selection_oof_auprc": score["auprc"],
                "nested_selection_oof_auprc_ci95_low": ci["auprc"][0],
                "nested_selection_oof_auprc_ci95_high": ci["auprc"][1],
                "nested_selection_oof_brier": score["brier_score"],
                "nested_selection_oof_brier_ci95_low": ci["brier_score"][0],
                "nested_selection_oof_brier_ci95_high": ci["brier_score"][1],
                "outer_fold_lr_selected_n": int(frequency.get("Logistic Regression", 0)),
                "outer_fold_xgb_selected_n": int(frequency.get("XGBoost", 0)),
                "training_oof_lr_auroc": lr_auc,
                "training_oof_xgb_auroc": xgb_auc,
                "training_oof_xgb_minus_lr_auroc": xgb_auc - lr_auc,
                "training_only_recommended_model": recommended,
                "previously_reported_model": REPORTED_MODEL[landmark],
                "recommendation_concordant_with_reported": recommended == REPORTED_MODEL[landmark],
                "locked_test_rows_used_for_selection": 0,
                "subject_overlap_any_fold": 0,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    inner_df.to_csv(OUT / "audit_v31_inner_model_selection.csv", index=False)
    outer_df.to_csv(OUT / "audit_v31_nested_outer_fold_performance.csv", index=False)
    nested_df.to_csv(OUT / "model_v31_nested_oof_predictions.csv", index=False)
    candidate_df.to_csv(OUT / "audit_v31_training_only_candidate_oof.csv", index=False)
    candidate_pred_df.to_csv(OUT / "model_v31_training_only_candidate_oof_predictions.csv", index=False)
    summary_df.to_csv(OUT / "audit_v31_training_only_model_selection_summary.csv", index=False)

    checks = {
        "original_split_subject_exclusive": bool(subject_split_n.max() == 1),
        "locked_test_rows_used_for_selection": 0,
        "all_inner_group_overlap_zero": bool(inner_df["inner_group_overlap_n"].eq(0).all()),
        "all_outer_group_overlap_zero": bool(outer_df["outer_group_overlap_n"].eq(0).all()),
        "nested_oof_unique_stays": bool(not nested_df[["landmark_hours", "stay_id"]].duplicated().any()),
        "landmarks_complete": sorted(summary_df["landmark_hours"].astype(int).tolist()) == LANDMARKS,
        "models_compared": MODELS,
        "selection_metric": "pooled inner out-of-fold AUROC",
        "hyperparameter_tuning": False,
    }
    (OUT / "audit_v31_validation.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    if not all(
        value
        for key, value in checks.items()
        if key not in {"models_compared", "selection_metric", "hyperparameter_tuning", "locked_test_rows_used_for_selection"}
    ):
        raise AssertionError(f"Validation failed: {checks}")

    lines = [
        "# v31 training-only nested grouped model-selection audit",
        "",
        "The locked 20% subject-level test partition was not used for model-family selection or cross-validation.",
        "Five outer and four inner stratified subject-grouped folds were used. Logistic regression and XGBoost retained their v27 preprocessing and fixed hyperparameters; no hyperparameter search was performed. Within each outer fold, the model family with the highest pooled inner out-of-fold AUROC was selected.",
        "",
        "## Results",
        "",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"- {int(row.landmark_hours)} h: nested-selection OOF AUROC {row.nested_selection_oof_auroc:.3f} "
            f"(95% CI {row.nested_selection_oof_auroc_ci95_low:.3f}-{row.nested_selection_oof_auroc_ci95_high:.3f}); "
            f"LR/XGBoost were selected in {int(row.outer_fold_lr_selected_n)}/{int(row.outer_fold_xgb_selected_n)} outer folds. "
            f"Full-training grouped OOF AUROC was {row.training_oof_lr_auroc:.3f} for LR and {row.training_oof_xgb_auroc:.3f} for XGBoost; "
            f"the training-only recommendation was {row.training_only_recommended_model}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This analysis audits the stability and leakage control of model-family selection. It estimates the internal performance of the selection procedure within the training partition; it does not replace the separately reported locked-test evaluation unless the manuscript explicitly adopts the training-only recommendation.",
        ]
    )
    (OUT / "audit_v31_results_brief.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
